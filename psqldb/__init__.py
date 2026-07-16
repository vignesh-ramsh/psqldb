"""
psqldb — ARC provider plugin: PostgreSQL via asyncpg.

Exports `arc.psqldb`: a connection pool plus the small set of primitives
that Relay's future bounded query engine (Architecture §3.4) will build on
— acquire/transaction/execute/fetch. Deliberately NOT a query builder or an
ORM: the bounded query engine and the raw-SQL escape hatch (`arc.sql(...)`)
both belong to Relay, not here. This plugin's whole job stops at "give me a
connection, safely, with pooling".

Lifecycle note (a known simplification, not an oversight): register() only
*constructs* the provider — it does not open the pool, because register()
itself is synchronous (arc.boot() is deliberately sync-callable) while
asyncpg.create_pool() is async. The application's own async entrypoint is
expected to `await arc.psqldb.open()` on startup and `await arc.psqldb
.close()` on shutdown. Automatic lifecycle wiring belongs to a future
Gateway/arc.health design, not to this plugin.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from . import fields, migrate, validation  # noqa: F401 - re-exported as arc.psqldb.fields / .migrate / .validation
from .model import SchemaError, TableSchema, load_patches_dir, load_schemas_dir  # noqa: F401
from .validation import ValidationError  # noqa: F401

CAPABILITY = "psqldb"

DSN_KEY = "psqldb_dsn"
POOL_MIN_SIZE_KEY = "psqldb_pool_min_size"
POOL_MAX_SIZE_KEY = "psqldb_pool_max_size"

# Only the fields the DB (or the CRUD helpers themselves) always supplies a
# correct value for get silently stripped from a caller's insert()/update()
# payload. `parent` and `idx` are also system-injected DDL (every child
# table gets them automatically), but neither has a value the database can
# invent on its own — `parent` is required with no default (which row this
# child belongs to) — so both must stay caller-settable.
_SYSTEM_COLUMN_NAMES = frozenset({"id", "created_at", "updated_at", "created_by", "updated_by", "_state"})


class PsqlDbProvider:
    """Thin asyncpg wrapper: a pool, the schema/migration system
    (register_model/plan/migrate), and the small set of CRUD primitives
    Relay's future bounded query engine will build on top of — see
    docs/arc.MD §3.4. Deliberately not a query builder or an ORM."""

    def __init__(self, kernel: Any, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        self._kernel = kernel
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: asyncpg.Pool | None = None
        self._schemas: list[TableSchema] = []
        self._by_table: dict[str, TableSchema] = {}
        self._patches: list[TableSchema] = []

    # ------------------------------------------------------------------ #
    # Schema registration — called from a business plugin's own
    # register(kernel), e.g. `arc.psqldb.register_model(Path(__file__).parent / "schemas")`.
    # Attributed to whichever plugin is currently registering (Kernel
    # tracks this already — see Kernel.current_plugin()) so callers never
    # pass their own name by hand.
    # ------------------------------------------------------------------ #
    def register_model(self, schemas_dir: str | Path) -> list[TableSchema]:
        plugin = self._kernel.current_plugin() or "<direct>"
        schemas = load_schemas_dir(Path(schemas_dir), plugin=plugin)
        for schema in schemas:
            if schema.table in self._by_table:
                raise SchemaError(
                    f"table '{schema.table}' is declared by both plugin "
                    f"'{self._by_table[schema.table].plugin}' and '{plugin}' — "
                    f"table names must be globally unique (mirrors the kernel's "
                    f"own duplicate-capability-name rule, §3.1)."
                )
            self._by_table[schema.table] = schema
        self._schemas.extend(schemas)
        return schemas

    def schemas(self) -> list[TableSchema]:
        return list(self._schemas)

    # ------------------------------------------------------------------ #
    # Patch registration — `arc.psqldb.register_patches(Path(__file__).parent / "patches")`.
    # Unlike register_model(), no table-name collision check happens here:
    # multiple plugins legitimately targeting the same table via patches is
    # the whole point. Field-id-level ownership collisions are caught later,
    # at plan/migrate time (psqldb.migrate — needs a DB round-trip to know
    # who owns what), not here.
    # ------------------------------------------------------------------ #
    def register_patches(self, patches_dir: str | Path) -> list[TableSchema]:
        plugin = self._kernel.current_plugin() or "<direct>"
        patches = load_patches_dir(Path(patches_dir), plugin=plugin)
        self._patches.extend(patches)
        return patches

    def patches(self) -> list[TableSchema]:
        return list(self._patches)

    def schema(self, table: str) -> TableSchema:
        """The table's CURRENT shape — its own schema's fields plus every
        patch registered against it, merged (a patch redeclaring an
        existing field id — a legitimate rename/retype of a field it owns —
        supersedes the schema's own version; a patch's NEW field ids are
        appended). This is "what fields actually exist on this table right
        now", which is what every caller outside the migration system
        wants — CRUD validation (validate_row/validate_columns_known)
        and the Query Engine (relay.query) both need to know about a
        patch-added column exactly the same way they know about a
        schema-declared one, or a real column becomes invisible to
        filters/fields/order_by while still being perfectly insertable
        (found via manual testing against a real Postgres — a patch-added
        REFERENCE column worked for insert/update, since those build SQL
        from the payload's own keys, but every Query Engine lookup that
        validates a field name against schema.column_fields() rejected it
        as "unknown").

        psqldb.migrate deliberately does NOT go through this — the
        ownership-scoped diffing in _diff_table needs schema and patches
        kept SEPARATE (self._schemas / self._patches), and works from
        those directly."""
        try:
            base = self._by_table[table]
        except KeyError:
            raise SchemaError(
                f"no registered schema for table '{table}' "
                f"(registered: {sorted(self._by_table) or 'none'})."
            ) from None
        patch_fields = [f for p in self._patches if p.table == table for f in p.fields]
        if not patch_fields:
            return base
        merged: dict[str, Any] = {f.id: f for f in base.fields}
        for f in patch_fields:
            merged[f.id] = f
        from dataclasses import replace
        return replace(base, fields=list(merged.values()))

    def ref_targets(self) -> dict[str, str]:
        return migrate.resolve_ref_targets([*self._schemas, *self._patches])

    def ref_columns(self) -> dict[tuple[str, str], migrate.ddl.RefColumn]:
        """(owning table, field name) -> resolved target (table, column,
        real SQL type) for every REFERENCE field — see
        migrate.resolve_ref_columns. Recomputed on every call, same as
        ref_targets() above (schemas don't change after boot, so this is
        cheap and never worth caching)."""
        return migrate.resolve_ref_columns([*self._schemas, *self._patches], self.ref_targets())

    async def open(self) -> None:
        """Create the pool. Idempotent — safe to call more than once.
        Registers a json/jsonb codec on every connection so JSON-typed
        columns (business JSON fields, and _trash.snapshot) round-trip as
        Python dict/list — asyncpg returns raw text for json/jsonb by
        default, which is surprising for every caller of insert()/get()."""
        import json as _json

        async def _init_connection(conn: asyncpg.Connection) -> None:
            await conn.set_type_codec(
                "jsonb", encoder=_json.dumps, decoder=_json.loads, schema="pg_catalog"
            )
            await conn.set_type_codec(
                "json", encoder=_json.dumps, decoder=_json.loads, schema="pg_catalog"
            )

        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn, min_size=self.min_size, max_size=self.max_size, init=_init_connection
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def acquire(self):
        """`async with arc.psqldb.acquire() as conn:` — one pooled connection."""
        if self._pool is None:
            raise RuntimeError(
                "psqldb pool is not open — call `await arc.psqldb.open()` "
                "during your application's startup first."
            )
        return self._pool.acquire()

    async def execute(self, query: str, *params: Any) -> str:
        async with self.acquire() as conn:
            return await conn.execute(query, *params)

    async def fetch(self, query: str, *params: Any) -> list[asyncpg.Record]:
        async with self.acquire() as conn:
            return await conn.fetch(query, *params)

    async def fetch_one(self, query: str, *params: Any) -> asyncpg.Record | None:
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *params)

    async def fetch_val(self, query: str, *params: Any) -> Any:
        async with self.acquire() as conn:
            return await conn.fetchval(query, *params)

    async def health(self) -> dict:
        try:
            version = await self.fetch_val("select version()")
            return {"ok": True, "version": version}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @contextlib.asynccontextmanager
    async def _conn_or(self, conn: Any):
        """Every CRUD primitive below accepts an optional `conn` — when
        given (Relay, running hooks + the write in one shared transaction),
        it's used directly and the caller owns the transaction; when None
        (direct psqldb use, unrelated to Relay), a connection is acquired
        and released here exactly as before."""
        if conn is not None:
            yield conn
        else:
            async with self.acquire() as c:
                yield c

    # ------------------------------------------------------------------ #
    # CRUD primitives — enough to exercise real tables now, ahead of Relay.
    # Relay's future engine wraps these with hooks/RBAC/query bounds; it
    # doesn't reinvent field validation or the soft-delete contract below.
    # ------------------------------------------------------------------ #
    async def insert(self, table: str, data: dict[str, Any], *, created_by: UUID | None = None, conn: Any = None) -> asyncpg.Record:
        schema = self.schema(table)
        clean = {k: v for k, v in data.items() if k not in _SYSTEM_COLUMN_NAMES}
        validation.validate_row(schema, clean)
        validation.validate_columns_known(schema, clean)

        async with self._conn_or(conn) as c:
            columns = list(clean.keys())
            if created_by is not None:
                columns.append("created_by")
                clean = {**clean, "created_by": created_by}
            placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
            col_list = ", ".join(f'"{c}"' for c in columns)
            query = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) RETURNING *'
            try:
                return await c.fetchrow(query, *(clean[col] for col in columns))
            except asyncpg.ForeignKeyViolationError as exc:
                raise validation.friendly_fk_error(exc, table=table) from exc
            except asyncpg.UniqueViolationError as exc:
                raise validation.friendly_unique_error(exc, table=table) from exc

    async def update(self, table: str, id: UUID, data: dict[str, Any], *, updated_by: UUID | None = None, conn: Any = None) -> asyncpg.Record | None:
        schema = self.schema(table)
        clean = {k: v for k, v in data.items() if k not in _SYSTEM_COLUMN_NAMES}
        validation.validate_row(schema, clean)
        validation.validate_columns_known(schema, clean)

        async with self._conn_or(conn) as c:
            columns = list(clean.keys())
            if updated_by is not None:
                columns.append("updated_by")
                clean = {**clean, "updated_by": updated_by}
            set_clause = ", ".join(f'"{col}" = ${i + 2}' for i, col in enumerate(columns))
            query = f'UPDATE "{table}" SET {set_clause} WHERE id = $1 RETURNING *'
            try:
                return await c.fetchrow(query, id, *(clean[col] for col in columns))
            except asyncpg.ForeignKeyViolationError as exc:
                raise validation.friendly_fk_error(exc, table=table) from exc
            except asyncpg.UniqueViolationError as exc:
                raise validation.friendly_unique_error(exc, table=table) from exc

    async def soft_delete(self, table: str, id: UUID, *, deleted_by: UUID | None = None, conn: Any = None) -> None:
        """Never a hard DELETE for a normal/child table — sets `_state = 99`.
        The DB-level arc_soft_delete_to_trash trigger (psqldb.ddl) takes it
        from there: snapshots the row into `_trash`, cascades to any child
        tables via the same mechanism, then physically removes the row.

        A `"system": true` table (psqldb.model) has none of that machinery —
        no `_state` column, no trigger attached (§3.9) — so soft-delete
        genuinely isn't applicable to it, only a real hard DELETE is. Not an
        error case: a system table simply IS the "always hard delete" table
        kind, same as `_trash`/`_field_registry`/`_patch_history` themselves
        would be if anything ever deleted from those. `_users`/`_access_keys`
        still get an audit-trail row for the delete via their own
        `arc_audit_{plugin}` trigger (fires on DELETE too), same as any other
        audited table — only `_trash` recovery doesn't apply."""
        schema = self.schema(table)  # raises SchemaError with a clear message if unknown
        async with self._conn_or(conn) as c:
            if schema.system:
                await c.execute(f'DELETE FROM "{table}" WHERE id = $1', id)
            else:
                await c.execute(
                    f'UPDATE "{table}" SET _state = 99, updated_by = $2 WHERE id = $1', id, deleted_by
                )

    # ------------------------------------------------------------------ #
    # Batch primitives — one multi-row SQL statement each, not a Python
    # loop over the single-row versions above. Relay's create_many/
    # update_many wrap these with per-row hooks (see relay's own module
    # docstring on why per-row, not per-batch).
    #
    # Deliberate constraint: every row/entry in one batch call must
    # provide the SAME set of fields. This isn't a shortcut — it avoids a
    # real correctness trap: a multi-row VALUES(...) list needs one fixed
    # column list, so a row missing a field would need either (a) an
    # explicit NULL, which would silently override that column's DB
    # DEFAULT for just that row, or (b) a second differently-shaped SQL
    # statement per distinct field-set. Neither is worth the complexity
    # for a first cut — callers with heterogeneous rows split the batch.
    # ------------------------------------------------------------------ #
    def _require_homogeneous(self, label: str, key_sets: list[set[str]]) -> list[str]:
        if not key_sets:
            return []
        columns = key_sets[0]
        for i, keys in enumerate(key_sets):
            if keys != columns:
                raise ValueError(
                    f"{label} requires every row to provide the same fields — "
                    f"row 0 has {sorted(columns)}, row {i} has {sorted(keys)}."
                )
        return sorted(columns)

    async def insert_many(self, table: str, rows: list[dict[str, Any]], *, created_by: UUID | None = None, conn: Any = None) -> list[asyncpg.Record]:
        if not rows:
            return []
        schema = self.schema(table)
        cleaned = []
        for data in rows:
            clean = {k: v for k, v in data.items() if k not in _SYSTEM_COLUMN_NAMES}
            validation.validate_row(schema, clean)
            validation.validate_columns_known(schema, clean)
            cleaned.append(clean)
        columns = self._require_homogeneous("insert_many", [set(c) for c in cleaned])
        if created_by is not None:
            columns = [*columns, "created_by"]
            cleaned = [{**c, "created_by": created_by} for c in cleaned]

        async with self._conn_or(conn) as c:
            col_list = ", ".join(f'"{col}"' for col in columns)
            value_rows, params, pi = [], [], 1
            for clean in cleaned:
                value_rows.append(f"({', '.join(f'${pi + j}' for j in range(len(columns)))})")
                params.extend(clean[col] for col in columns)
                pi += len(columns)
            query = f'INSERT INTO "{table}" ({col_list}) VALUES {", ".join(value_rows)} RETURNING *'
            try:
                return await c.fetch(query, *params)
            except asyncpg.ForeignKeyViolationError as exc:
                raise validation.friendly_fk_error(exc, table=table) from exc
            except asyncpg.UniqueViolationError as exc:
                raise validation.friendly_unique_error(exc, table=table) from exc

    async def update_many(self, table: str, updates: list[dict[str, Any]], *, updated_by: UUID | None = None, conn: Any = None) -> list[asyncpg.Record]:
        """`updates` is `[{"id": ..., "data": {...}}, ...]` — every entry's
        `data` must update the same fields, same homogeneity rule as
        insert_many."""
        if not updates:
            return []
        schema = self.schema(table)
        ids, cleaned = [], []
        for i, u in enumerate(updates):
            if "id" not in u or "data" not in u:
                raise ValueError(f"update_many: entry {i} must have 'id' and 'data' keys.")
            clean = {k: v for k, v in u["data"].items() if k not in _SYSTEM_COLUMN_NAMES}
            validation.validate_row(schema, clean)
            validation.validate_columns_known(schema, clean)
            ids.append(u["id"])
            cleaned.append(clean)
        columns = self._require_homogeneous("update_many", [set(c) for c in cleaned])
        if updated_by is not None:
            columns = [*columns, "updated_by"]
            cleaned = [{**c, "updated_by": updated_by} for c in cleaned]

        async with self._conn_or(conn) as c:
            # Every value in an anonymous VALUES(...) list needs an explicit
            # cast — Postgres plans a prepared statement before it ever sees
            # the actual parameter values, so it can't infer "this matches
            # the numeric salary column" on its own; it silently falls back
            # to `text`, and every later reference to that column (in WHERE,
            # in SET) fails with a type mismatch. Casting per-value here,
            # right where each one enters the query, is what actually fixes
            # it — casting only the WHERE/SET reference isn't enough once a
            # second, differently-typed column is involved.
            #
            # f.sql_type() itself is wrong for a REFERENCE field whose
            # target_field points at a non-"id" column — it always returns
            # "UUID" (Field.sql_type() only knows what ONE field alone can
            # determine; see its docstring), but such a field's real
            # physical type could be e.g. VARCHAR(8). Casting a business-key
            # string to ::UUID here would fail outright, so REFERENCE columns
            # go through the cross-schema-resolved type instead.
            ref_columns = self.ref_columns()
            field_types = {
                f.name: (ref_columns[(table, f.name)].sql_type if f.type == "REFERENCE" else f.sql_type())
                for f in schema.fields if f.is_column()
            }
            data_columns = ["_row_id", *columns]
            data_casts = ["uuid", *(field_types[col] for col in columns)]
            set_clause = ", ".join(f'"{col}" = data."{col}"' for col in columns)
            data_col_list = ", ".join(f'"{col}"' for col in data_columns)
            value_rows, params, pi = [], [], 1
            for row_id, clean in zip(ids, cleaned):
                values = [row_id, *(clean[col] for col in columns)]
                value_rows.append(
                    "(" + ", ".join(f"${pi + j}::{data_casts[j]}" for j in range(len(data_columns))) + ")"
                )
                params.extend(values)
                pi += len(data_columns)
            query = (
                f'UPDATE "{table}" SET {set_clause} '
                f'FROM (VALUES {", ".join(value_rows)}) AS data({data_col_list}) '
                f'WHERE "{table}".id = data._row_id RETURNING "{table}".*'
            )
            try:
                return await c.fetch(query, *params)
            except asyncpg.ForeignKeyViolationError as exc:
                raise validation.friendly_fk_error(exc, table=table) from exc
            except asyncpg.UniqueViolationError as exc:
                raise validation.friendly_unique_error(exc, table=table) from exc

    async def soft_delete_many(self, table: str, ids: list[UUID], *, deleted_by: UUID | None = None, conn: Any = None) -> None:
        if not ids:
            return
        schema = self.schema(table)
        async with self._conn_or(conn) as c:
            if schema.system:
                await c.execute(f'DELETE FROM "{table}" WHERE id = ANY($1::uuid[])', ids)
            else:
                await c.execute(
                    f'UPDATE "{table}" SET _state = 99, updated_by = $1 WHERE id = ANY($2::uuid[])',
                    deleted_by, ids,
                )

    async def get_many(self, table: str, ids: list[UUID], *, conn: Any = None) -> list[asyncpg.Record]:
        self.schema(table)
        if not ids:
            return []
        async with self._conn_or(conn) as c:
            return await c.fetch(f'SELECT * FROM "{table}" WHERE id = ANY($1::uuid[])', ids)

    async def clear(self, table: str, *, cleared_by: UUID | None = None) -> int:
        """`arc psqldb clear` — every row in `table` goes through the same
        soft-delete-to-trash trigger as a single soft_delete() would, not a
        raw TRUNCATE: recoverable from _trash, and cascades to child tables
        automatically (same mechanism, no special-casing here). Slower than
        TRUNCATE for a very large table (one trigger firing per row) —
        deliberate, matching every other destructive path in this system.
        A `"system": true` table has no _state/trigger machinery (see
        soft_delete's docstring) — cleared via a real DELETE instead, same
        "always hard delete" exception. Returns the number of rows cleared."""
        schema = self.schema(table)  # raises SchemaError with a clear message if unknown
        if schema.system:
            result = await self.execute(f'DELETE FROM "{table}"')
        else:
            result = await self.execute(
                f'UPDATE "{table}" SET _state = 99, updated_by = $1 WHERE _state IS DISTINCT FROM 99',
                cleared_by,
            )
        return int(result.split()[-1]) if result else 0

    # ------------------------------------------------------------------ #
    # `arc plugin disable <name> --wipe` — DDL-level, irreversible (unlike
    # clear() above, there's no _trash for a dropped TABLE, only for a
    # dropped ROW). Two real dependency questions, checked two different
    # ways because only one of them is something Postgres itself knows
    # about:
    #   1. Did another plugin PATCH extra fields onto a table this plugin
    #      owns? Postgres has no concept of "which ARC plugin owns this
    #      column" — dropping the table trivially takes those columns with
    #      it and Postgres raises nothing. Checked ourselves, against
    #      _field_registry, before touching anything.
    #   2. Does some OTHER plugin's table have a live REFERENCE (a real
    #      FK) pointing at one of these? Postgres already enforces this —
    #      a plain DROP TABLE (no CASCADE) fails outright with
    #      DependentObjectsStillExistError if so. Caught, not
    #      pre-checked ourselves — same "let the real constraint be the
    #      safety net" posture as the FK-existence pre-check that used to
    #      exist on the write path and was removed for being redundant
    #      (§3.9's write-path hardening notes).
    # Both are refused by default; `force=True` overrides both — dropping
    # the table anyway (patch case) or retrying with CASCADE (FK case).
    # ------------------------------------------------------------------ #
    async def wipe_plugin_tables(self, plugin: str, *, force: bool = False, dry_run: bool = False) -> dict:
        """Returns {"tables": [...ordered...], "audit_table": str | None,
        "row_counts": {table: int}}. `row_counts` is always computed (even
        on a real, non-dry-run call) so the caller can show what was about
        to be destroyed either way. `dry_run=True` stops after computing
        the plan — no DDL executed at all, safe to call purely to preview.
        Raises MigrationError (never touches anything) if another plugin
        owns fields on one of these tables and force=False."""
        tables = {s.table for s in self.schemas() if s.plugin == plugin}
        if not tables:
            return {"tables": [], "audit_table": None, "row_counts": {}}

        if not force:
            foreign = await self.fetch(
                'SELECT DISTINCT "table", plugin FROM _field_registry '
                'WHERE "table" = ANY($1) AND plugin != $2',
                list(tables), plugin,
            )
            if foreign:
                detail = ", ".join(f"'{r['table']}' (has fields owned by '{r['plugin']}')" for r in foreign)
                raise migrate.MigrationError(
                    f"cannot wipe '{plugin}': {detail} — another plugin has patched fields onto "
                    f"a table '{plugin}' owns; dropping it would destroy that plugin's columns "
                    f"and data too. Pass force=True to do it anyway."
                )

        ordered = migrate.order_for_clear(tables, self.ref_columns()) if len(tables) > 1 else sorted(tables)

        audit_table = f"_audit_{plugin}"
        exists = await self.fetch_val(
            "SELECT 1 FROM information_schema.tables WHERE table_name = $1", audit_table
        )
        audit_table = audit_table if exists else None
        all_tables = [*ordered, *([audit_table] if audit_table else [])]

        row_counts = {t: await self.fetch_val(f'SELECT COUNT(*) FROM "{t}"') for t in all_tables}

        if dry_run:
            return {"tables": ordered, "audit_table": audit_table, "row_counts": row_counts}

        for table in all_tables:
            await self._drop_table_for_wipe(table, force=force)
        if audit_table:
            # The trigger itself disappears along with its table (a
            # trigger is a property of the table it's attached to) — only
            # the shared trigger FUNCTION is a separate object that
            # outlives it and needs its own cleanup.
            await self.execute(f'DROP FUNCTION IF EXISTS "arc_audit_{plugin}"() CASCADE')

        await self.execute('DELETE FROM _field_registry WHERE "table" = ANY($1)', all_tables)
        await self.execute('DELETE FROM _patch_history WHERE "table" = ANY($1)', all_tables)

        return {"tables": ordered, "audit_table": audit_table, "row_counts": row_counts}

    async def _drop_table_for_wipe(self, table: str, *, force: bool) -> None:
        try:
            await self.execute(f'DROP TABLE "{table}"')
        except asyncpg.exceptions.DependentObjectsStillExistError as exc:
            if not force:
                raise migrate.MigrationError(
                    f"cannot drop '{table}': something outside this plugin still references it "
                    f"({exc}). Pass force=True to CASCADE — this destroys the dependent object too."
                ) from exc
            await self.execute(f'DROP TABLE "{table}" CASCADE')

    async def get(self, table: str, id: UUID, *, conn: Any = None) -> asyncpg.Record | None:
        self.schema(table)
        async with self._conn_or(conn) as c:
            return await c.fetchrow(f'SELECT * FROM "{table}" WHERE id = $1', id)

    async def get_by(self, table: str, filters: dict[str, Any], *, conn: Any = None) -> asyncpg.Record | None:
        """Single-row lookup by any field(s), not just id — equality only,
        first match. A deliberately minimal slice of the still-undesigned
        bounded Query Engine (§3.4): built because a real need for it
        surfaced (looking a row up by a business key, not its UUID), not a
        first piece of that larger, separate system."""
        schema = self.schema(table)
        if not filters:
            raise ValueError("get_by() requires at least one filter.")
        known = {f.name for f in schema.all_fields() if f.is_column()}
        unknown = [k for k in filters if k not in known]
        if unknown:
            raise ValueError(f"get_by(): unknown field(s) {unknown} on table '{table}'.")
        async with self._conn_or(conn) as c:
            where = " AND ".join(f'"{k}" = ${i + 1}' for i, k in enumerate(filters))
            query = f'SELECT * FROM "{table}" WHERE {where} LIMIT 1'
            return await c.fetchrow(query, *filters.values())


def register(kernel: Any) -> None:
    kernel.settings.declare(DSN_KEY, secret=True)
    kernel.settings.declare(POOL_MIN_SIZE_KEY)
    kernel.settings.declare(POOL_MAX_SIZE_KEY)

    dsn = kernel.settings.get(DSN_KEY, reveal=True)
    if dsn is None:
        raise RuntimeError(
            f"'{DSN_KEY}' is not set. Run: "
            f"arc settings set {DSN_KEY} postgresql://user:pass@host:5432/dbname --secret"
        )

    min_size = int(kernel.settings.get(POOL_MIN_SIZE_KEY) or 1)
    max_size = int(kernel.settings.get(POOL_MAX_SIZE_KEY) or 10)

    provider = PsqlDbProvider(kernel, dsn, min_size=min_size, max_size=max_size)
    kernel.export(CAPABILITY, provider, requires=[], optional_requires=[])