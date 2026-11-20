"""
pgdb.migrate
-------------------
Orchestrates everything `arc pgdb plan` / `arc pgdb migrate` do:
collect declared schemas AND patches -> diff each against `_field_registry`
(the DB's memory of "the schema as of the last successful migration", see
pgdb.introspect) -> render a plan -> optionally apply it, updating
`_field_registry` to match and recording the change in `_patch_history`.

Two declaration sources, one diff engine:
  * schemas/<Table>.json — a plugin fully owns and creates this table.
  * patches/<Table>.json — a plugin adds fields to / modifies fields it
    already owns on a table that already exists (possibly owned by a
    DIFFERENT plugin's schema). Never creates a table (skipped with a
    warning if the target doesn't exist yet, or if the target is a
    `"system": true` table — those self-declare their entire structure,
    pgdb.model, so no other plugin's patch ever applies to one); can drop
    a field it previously added via an earlier patch if that patch no
    longer declares it — same Trash-backed path as any other column drop.
  * Both are diffed OWNERSHIP-SCOPED: a schema or patch only ever compares
    against the _field_registry rows IT owns on that table (`plugin` column),
    never another plugin's — otherwise plugin A's unrelated re-migrate would
    see plugin B's patched-in field as "missing from my declaration" and
    try to drop or rename it back. Attempting to ADD a field id that
    already belongs to a DIFFERENT plugin, or declaring a whole table that
    already physically exists under a different (or no) owner, is a hard
    error raised BEFORE any SQL runs — the existing table/field is always
    left untouched, never dropped or silently adopted.

Deliberate simplifications (not oversights — narrower than the full design
on purpose, safe to widen later):
  * child tables may not themselves declare a TABLE field (no nested
    children yet) — validated, hard error if violated.
  * named indexes are created with a plain CREATE INDEX inside the migration
    transaction (never CONCURRENTLY, which cannot run in one), so they block
    writes to the table while they build.
  * a table rename is not automated: a renamed schema file is a new table
    plus a table that is no longer declared (reported, not dropped, unless
    --allow-drops is given).

What never happens just because something is not declared any more: a table,
or the columns of a deleted patch file, are NOT dropped unless the operator
passes allow_drops (`arc pgdb migrate --allow-drops`). Everything that is
dropped, and every column type change, copies its values to _trash first.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import heapq
import json
import logging
import re
import zlib
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Literal

import arc

from . import ddl, introspect
from .fields import Field, FieldError
from .model import SchemaError, TableSchema, load_schemas_dir

_logger = logging.getLogger("pgdb")

# One fixed key shared by every `arc pgdb migrate` run against every
# database — a stable (deterministic, not Python's salted hash()) 32-bit
# value derived from a fixed string, well within Postgres's signed-bigint
# advisory-lock key range. Not a real secret, not table/schema-specific:
# the whole point is that ANY two migrate runs against the SAME database
# contend for the SAME lock, regardless of which plugin/table triggered
# them.
MIGRATE_LOCK_KEY = zlib.crc32(b"arc_psqldb_migrate")

#: How long one DDL statement of a migration may wait for its table lock before
#: the migration fails (and rolls back) instead of freezing the table's traffic.
MIGRATION_LOCK_TIMEOUT = "30s"

OpKind = Literal[
    "create_table",
    "drop_table",
    "add_column",
    "drop_column",
    "rename_column",
    "alter_type",
    "set_not_null",
    "drop_not_null",
    "add_unique",
    "drop_unique",
    "ensure_index",
    "create_index",
    "drop_index",
    "drop_trigger",
    "set_default",
    "drop_default",
    "add_foreign_key",
    "drop_foreign_key",
    "add_unique_together",
    "drop_unique_together",
]
OpSource = Literal["schema", "patch"]


@dataclass
class Op:
    kind: OpKind
    table: str
    plugin: str
    description: str  # human-readable, shown in `plan`/`migrate`
    sql: list[str]  # statement(s) that implement this op
    destructive: bool = False
    source: OpSource = "schema"
    # True for exactly 4 append sites in build_plan: the shared trigger-
    # function CREATE OR REPLACE, "ensure audit table exists", "ensure
    # index", and "attach audit trigger" — all idempotent SQL, appended
    # UNCONDITIONALLY on every single build_plan() call regardless of
    # whether anything actually changed (deliberately: this is the
    # self-healing mechanism that re-establishes a trigger/index someone
    # dropped by hand, with no diff needed to detect the drift). Still
    # always applied — this flag only controls what plan/migrate DISPLAY
    # and whether is_empty()-for-humans (has_real_changes(), below)
    # counts them, never whether they run.
    maintenance: bool = False


@dataclass
class MigrationPlan:
    ops: list[Op] = dc_field(default_factory=list)
    schemas: list[TableSchema] = dc_field(
        default_factory=list
    )  # schemas + patches this plan covers
    ref_targets: dict[str, str] = dc_field(
        default_factory=dict
    )  # schema-stem -> physical table name
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"] = dc_field(
        default_factory=dict
    )  # (table, field name) -> resolved target
    warnings: list[str] = dc_field(default_factory=list)  # e.g. "skipped patch for missing table"

    def destructive_ops(self) -> list[Op]:
        return [op for op in self.ops if op.destructive]

    def by_table(self) -> dict[str, list[Op]]:
        out: dict[str, list[Op]] = {}
        for op in self.ops:
            out.setdefault(op.table, []).append(op)
        return out

    def real_ops(self) -> list[Op]:
        """`self.ops` minus the always-present maintenance ops (Op.maintenance's
        own docstring has the full list) — what a HUMAN should be shown as
        "the migration," as opposed to what actually gets executed (every op,
        maintenance included — see is_empty(), still used for that)."""
        return [op for op in self.ops if not op.maintenance]

    def has_real_changes(self) -> bool:
        """Unlike is_empty() (used by apply_plan callers to decide whether
        there's anything to EXECUTE — maintenance ops still count, since
        they still need to run), this is "is there anything a human should
        review or be told about" — the plan/migrate CLI's own question.
        A plan containing ONLY the 4 always-present maintenance ops
        (nothing genuinely changed) answers False here, even though
        is_empty() on that same plan is False (there ARE ops — they're
        just not a real diff)."""
        return any(not op.maintenance for op in self.ops)

    def is_empty(self) -> bool:
        return not self.ops


class MigrationError(RuntimeError):
    pass


@contextlib.asynccontextmanager
async def migration_lock(conn: Any, *, poll_interval: float = 1.0):
    """Serializes concurrent `arc pgdb migrate` runs against the same
    database — two replicas deploying at once must not both diff-and-apply
    DDL concurrently. Uses Postgres's own session-level advisory lock
    (pg_advisory_lock/pg_try_advisory_lock/pg_advisory_unlock): it works
    over the SAME connection already in use (no extra infrastructure, no
    redix dependency), and Postgres releases it automatically if this
    process or connection dies — no stale-lock cleanup to ever handle.

    Deliberately BLOCKS (polling pg_try_advisory_lock rather than the
    blocking pg_advisory_lock, so a friendly "waiting" message can be
    logged once rather than the caller just appearing to hang) instead of
    failing outright — "serialize", not "reject": the second replica
    should wait its turn, then build a plan against the now-already-
    migrated database and find nothing left to do, not error out just
    because it lost a race.

    IMPORTANT: this is a SESSION-level lock, tied to `conn` itself, not to
    the transaction migrate.apply_plan() opens inside it — returning `conn`
    to the pool afterward does NOT close the underlying connection (the
    pool reuses it), so the lock is only ever released by the explicit
    pg_advisory_unlock in `finally` below, never implicitly. Every caller
    MUST use this as a context manager (`async with migration_lock(conn):`)
    so that release always runs, including when the protected code raises.
    """
    announced = False
    while True:
        acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", MIGRATE_LOCK_KEY)
        if acquired:
            break
        if not announced:
            _logger.info(
                "another `arc pgdb migrate` is already running against this "
                "database — waiting for it to finish..."
            )
            announced = True
        await asyncio.sleep(poll_interval)
    try:
        yield
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", MIGRATE_LOCK_KEY)


def _q(value: Any) -> str:
    """SQL string-literal escaping for the bookkeeping statements below
    (registry/history upserts, trash snapshots) that are rendered as full
    text rather than parameterized. Doubling single quotes is sufficient
    for a standard-conforming-strings Postgres literal — without it, any
    schema value containing an apostrophe (e.g. a field default of
    \"O'Brien\") produced broken SQL and a failed migrate."""
    return str(value).replace("'", "''")


# ------------------------------------------------------------------------ #
# Ownership: who may change which table
# ------------------------------------------------------------------------ #
def refuse_patches_on_own_tables(schemas: list[TableSchema], patches: list[TableSchema]) -> None:
    """A plugin changes the tables it OWNS in their schema files, never with a patch. A patch is how one plugin adds
    fields to a table ANOTHER plugin owns; on a plugin's own table it only splits one table's shape over two files,
    so that what a table looks like depends on which of them you read (and on the order they merge in). Raises
    MigrationError naming the patch, the table and the schema file to move the fields into, before anything is
    planned, so no SQL runs. Ownership is "this plugin has a schema file for the table"; a patch from any other
    plugin is untouched by this."""
    owned = {(s.plugin, s.table): s for s in schemas}
    for patch in patches:
        schema = owned.get((patch.plugin, patch.table))
        if schema is None:
            continue
        raise MigrationError(
            f"patch '{patch.source_path.name}' (plugin '{patch.plugin}') targets table '{patch.table}', which plugin "
            f"'{patch.plugin}' owns itself (schemas/{schema.source_path.name}). A plugin changes its own tables in "
            f"their schema file, never with a patch: move the patch's fields and indexes into "
            f"schemas/{schema.source_path.name} (keep the same field ids, so nothing is altered in the database) and "
            f"delete the patch. Patches are for adding fields to another plugin's table."
        )


# ------------------------------------------------------------------------ #
# Ordering + relational resolution — schemas only; patches never create a
# table or own a child, so they never participate in this.
# ------------------------------------------------------------------------ #
def _resolve_child_owners(schemas: list[TableSchema]) -> dict[str, TableSchema]:
    """Every `child: true` schema must be targeted by exactly one TABLE
    field somewhere. Returns {child_schema_stem: owning TableSchema}.

    Also validates the REVERSE direction, which used to go entirely
    unchecked: a TABLE field's `target` must itself be a `child: true`
    schema. TABLE fields render no column at all on the owning table
    (Field.is_column() is False for TABLE), and a target schema that isn't
    `child: true` gets ordinary NORMAL_SYSTEM_FIELDS (no `parent` column
    either) — so a TABLE field pointing at a non-child schema used to be a
    completely silent no-op: no column, no FK, no error, no warning. A
    plugin author who simply forgot `"child": true` on the target got a
    schema that parsed fine and a relationship that did nothing."""
    by_stem = {s.source_path.stem: s for s in schemas}
    owners: dict[str, list[str]] = {}
    for s in schemas:
        for f in s.child_fields():
            target_schema = by_stem.get(f.target)
            # An unresolvable target (matches no schema file at all) is
            # left to resolve_ref_targets — called right after this in
            # build_plan — to raise its own "unknown target" error; this
            # function only adds the narrower check below, which fires
            # when the target DOES exist but isn't a child schema.
            if target_schema is not None and not target_schema.child:
                raise MigrationError(
                    f"schema '{s.table}', field '{f.name}' (type TABLE) targets "
                    f"'{f.target}' ('{target_schema.table}'), which does not declare "
                    f'"child": true — a TABLE field\'s target must be a child schema, '
                    f"or the relationship is silently a no-op (TABLE fields render no "
                    f"column on the owning table, and a non-child target gets no "
                    f"'parent' column either). Add \"child\": true to '{f.target}', or "
                    f"use type REFERENCE instead if you meant a plain link, not "
                    f"ownership."
                )
            owners.setdefault(f.target, []).append(s.table)

    for child in schemas:
        if not child.child:
            continue
        if child.child_fields():
            raise MigrationError(
                f"schema '{child.table}' is a child table but itself declares a "
                f"TABLE field — nested child tables aren't supported yet."
            )
        claimants = owners.get(child.source_path.stem, [])
        if not claimants:
            raise MigrationError(
                f"schema '{child.table}' has \"child\": true but no other schema's "
                f"TABLE field targets it — a child table must be owned by exactly "
                f"one parent."
            )
        if len(claimants) > 1:
            raise MigrationError(
                f"schema '{child.table}' is targeted by more than one TABLE field "
                f"({claimants}) — a child table can only have one owner."
            )
    return {stem: by_stem[stem] for stem in by_stem if by_stem[stem].child}


#  resolve_ref_targets's own "logged once, not every call" set for the
# table-name-spelling fallback below — module-level and never cleared, on
# purpose: the point is one notice per distinct `target` value for the
# lifetime of the process, not one per resolve_ref_targets() call (which
# may run repeatedly — plan, then migrate, then a system.reload
# recomputing pgdb's own cached ref_targets()).
_warned_table_name_targets: set[str] = set()


def resolve_ref_targets(schemas: list[TableSchema]) -> dict[str, str]:
    """Maps each REFERENCE/TABLE field's raw `target` to the physical,
    slugified table it resolves to (e.g. "Employee" -> "employee"). Used
    both for DDL rendering and for insert()-time FK existence checks
    (pgdb.validation). Callers pass schemas + patches together so a
    patch's own REFERENCE fields resolve against the same known-table set
    schemas do.

    `target` is conventionally a schema-file STEM ("Employee") — but a
    field naming the physical TABLE name directly ("employee") now also
    resolves, with one logged notice per distinct target the first time
    it's used that way, never a hard error: both spellings mean the exact
    same table (the physical name a schema file's own stem produces), so
    there's no real ambiguity to worry about — just a footgun the resolver
    can remove instead of leaving it to documentation."""
    by_stem = {s.source_path.stem: s for s in schemas}
    by_table = {s.table: s for s in schemas}
    ref_targets: dict[str, str] = {}
    for s in schemas:
        for f in s.fields:
            if f.type in ("REFERENCE", "TABLE") and f.target:
                target_schema = by_stem.get(f.target)
                if target_schema is None:
                    target_schema = by_table.get(f.target)
                    if target_schema is not None and f.target not in _warned_table_name_targets:
                        _warned_table_name_targets.add(f.target)
                        _logger.info(
                            f"schema '{s.table}', field '{f.name}': target '{f.target}' "
                            f"is a physical table name — resolved fine, but the "
                            f"conventional spelling is the schema file's own stem "
                            f"('{target_schema.source_path.stem}')."
                        )
                if target_schema is None:
                    raise MigrationError(
                        f"schema '{s.table}', field '{f.name}': target '{f.target}' "
                        f"does not match any known schema file name or table name."
                    )
                ref_targets[f.target] = target_schema.table
    return ref_targets


def resolve_ref_columns(
    schemas: list[TableSchema], ref_targets: dict[str, str]
) -> dict[tuple[str, str], "ddl.RefColumn"]:
    """Resolves every REFERENCE field's real target column — "id" (the
    default) or whatever `target_field` names — into a ddl.RefColumn
    carrying the target table, target column, and that column's own SQL
    type (the referencing column must be declared to match it; see
    Field.target_field and ddl.RefColumn).

    Keyed by (owning table, field NAME), not by target stem alone — unlike
    ref_targets, where every field pointing at the same target stem always
    resolves to the same table (so the stem alone is a safe key), two
    different REFERENCE fields can point at the same target table through
    two DIFFERENT columns (one via the default 'id', another via its own
    target_field) — collapsing them onto one key would silently drop one.

    Raises MigrationError, before any SQL runs, when:
      * target_field names a field that isn't declared on the target schema,
      * target_field names a field that isn't declared "unique": true —
        Postgres can only create a foreign key against a unique column, and
        this project would rather fail loudly here than let that surface as
        a raw DB error mid-migrate.
    Callers pass schemas + patches together, same as resolve_ref_targets —
    a patch can add a new REFERENCE field with its own target_field.

    `fields_by_stem` (not `{stem: schema}`, deliberately): a schema and a
    patch sharing the same file stem are BOTH present in `schemas` here,
    and a plain last-one-wins dict keyed by stem would silently pick
    whichever happened to come later in the list — the patch, in
    practice, since build_plan always orders `[*schemas, *patches]`. That
    made a target_field lookup search only the PATCH's own (typically
    tiny) field list, failing to find a field the real schema declares,
    with a confusing "does not match any field declared" error — found
    exactly this way, referencing a patched table's schema-declared field.
    Accumulating every schema/patch's fields under their shared stem finds
    a target_field regardless of which file actually declares it."""
    fields_by_stem: dict[str, list] = {}
    for s in schemas:
        fields_by_stem.setdefault(s.source_path.stem, []).extend(s.fields)
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"] = {}
    for s in schemas:
        for f in s.fields:
            if f.type != "REFERENCE" or not f.target:
                continue
            target_table = ref_targets[f.target]
            if f.target_field is None:
                ref_columns[(s.table, f.name)] = ddl.RefColumn(
                    table=target_table, column="id", sql_type="UUID"
                )
                continue
            target_field = next(
                (tf for tf in fields_by_stem.get(f.target, []) if tf.name == f.target_field), None
            )
            if target_field is None:
                raise MigrationError(
                    f"schema '{s.table}', field '{f.name}': target_field "
                    f"'{f.target_field}' does not match any field declared on "
                    f"target schema '{f.target}' (table '{target_table}')."
                )
            if not target_field.unique:
                raise MigrationError(
                    f"schema '{s.table}', field '{f.name}': target_field "
                    f"'{f.target_field}' on '{target_table}' is not declared "
                    f'"unique": true — Postgres can only create a foreign key '
                    f"against a unique column. Mark '{f.target_field}' "
                    f"\"unique\": true on its own schema ('{f.target}'), or remove "
                    f"target_field on '{s.table}'.'{f.name}' to reference the "
                    f"default 'id' primary key instead."
                )
            ref_columns[(s.table, f.name)] = ddl.RefColumn(
                table=target_table, column=target_field.name, sql_type=target_field.sql_type()
            )
    return ref_columns


def order_for_clear(
    tables: set[str], ref_columns: dict[tuple[str, str], "ddl.RefColumn"]
) -> list[str]:
    """Orders a set of tables so `arc pgdb clear -p <plugin>` (which
    clears every table a plugin owns in one pass) respects REFERENCE
    dependencies among them: a table that references another via a real
    FK (ON DELETE RESTRICT, §3.9's REFERENCE type) must be cleared FIRST —
    clear's soft-delete-to-trash trigger issues a real physical DELETE the
    moment a row's _state hits 99, and that DELETE fails outright with a
    live ForeignKeyViolationError if some OTHER row (in this same clear
    batch) still references it. Found by running `arc pgdb clear -p
    example_hr` for real: 'department' sorted alphabetically before
    'employee', but employee.department REFERENCES department.code —
    clearing department's rows first hit exactly that violation.

    Only considers REFERENCE edges where BOTH ends are inside `tables` — a
    reference to/from a table OUTSIDE this clear's scope isn't this
    function's problem (that table isn't being touched here at all).
    TABLE (child) relationships need no such ordering: those FKs are ON
    DELETE CASCADE, and the trash trigger's own recursive cascade already
    handles parent-before-child correctly regardless of iteration order.

    Raises MigrationError on a circular REFERENCE dependency within the
    set — genuinely unclearable together in any single pass; the caller
    has to break the cycle by hand (clear one side individually first)."""
    adjacency: dict[str, set[str]] = {t: set() for t in tables}
    for (owner, _field), ref in ref_columns.items():
        if owner in tables and ref.table in tables and ref.table != owner:
            adjacency[owner].add(ref.table)  # owner must be cleared before ref.table

    indegree = {t: 0 for t in tables}
    for outs in adjacency.values():
        for out in outs:
            indegree[out] += 1
    heap = sorted(t for t, d in indegree.items() if d == 0)
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        t = heapq.heappop(heap)
        order.append(t)
        for out in sorted(adjacency[t]):
            indegree[out] -= 1
            if indegree[out] == 0:
                heapq.heappush(heap, out)

    if len(order) != len(tables):
        remaining = sorted(set(tables) - set(order))
        raise MigrationError(
            f"cannot determine a safe clear order for {remaining} — a circular "
            f"REFERENCE dependency exists among them. Clear them individually "
            f"(`arc pgdb clear -t <table>`), breaking the cycle by hand first."
        )
    return order


def order_for_create(
    tables: list[str], ref_columns: dict[tuple[str, str], "ddl.RefColumn"]
) -> list[str]:
    """Orders a set of tables so a fresh `arc pgdb migrate` creates a
    REFERENCE's target table before the table that points to it — the
    opposite requirement from order_for_clear (§ above), same underlying
    dependency graph. ddl._user_column_sql renders REFERENCES inline in the
    CREATE TABLE statement itself, so creating the referencing table first
    fails outright with an UndefinedTableError the moment its target
    doesn't exist yet.

    Found the same way order_for_clear's bug was found: `_order_and_link`
    previously only ordered TABLE (child) schemas after their parents —
    plain REFERENCE dependencies among non-child tables were never
    considered at all, and schema files are loaded in sorted filename
    order (pgdb.model.load_schemas_dir), which has no reason to match
    dependency order. A schema named "_access_keys.json" referencing
    "_users.json" sorts before it alphabetically — exactly the failure
    case this closes.

    Only considers REFERENCE edges where BOTH ends are inside `tables` —
    matches order_for_clear's own scoping rule (a table already created in
    an earlier migrate run isn't this ordering's problem). Raises
    MigrationError on a circular REFERENCE dependency, same as
    order_for_clear."""
    table_set = set(tables)
    adjacency: dict[str, set[str]] = {t: set() for t in tables}
    for (owner, _field), ref in ref_columns.items():
        if owner in table_set and ref.table in table_set and ref.table != owner:
            adjacency[ref.table].add(owner)  # ref.table must be created before owner

    indegree = {t: 0 for t in tables}
    for outs in adjacency.values():
        for out in outs:
            indegree[out] += 1
    heap = sorted(t for t, d in indegree.items() if d == 0)
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        t = heapq.heappop(heap)
        order.append(t)
        for out in sorted(adjacency[t]):
            indegree[out] -= 1
            if indegree[out] == 0:
                heapq.heappush(heap, out)

    if len(order) != len(tables):
        remaining = sorted(set(tables) - set(order))
        raise MigrationError(
            f"cannot determine a safe create order for {remaining} — a circular "
            f"REFERENCE dependency exists among them."
        )
    return order


def _order_and_link(
    schemas: list[TableSchema],
    ref_targets: dict[str, str],
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"] | None = None,
) -> tuple[list[TableSchema], dict[str, str]]:
    """Returns (ordered schemas — parents before children, each bucket
    itself ordered by REFERENCE dependency via order_for_create so a fresh
    CREATE TABLE never references a target that doesn't exist yet; parent_of
    map "child table" -> "parent table")."""
    by_stem = {s.source_path.stem: s for s in schemas}
    parent_of: dict[str, str] = {}
    for s in schemas:
        for f in s.child_fields():
            child_schema = by_stem[f.target]
            parent_of[child_schema.table] = s.table

    parents = [s for s in schemas if not s.child]
    children = [s for s in schemas if s.child]

    if ref_columns:
        by_table = {s.table: s for s in schemas}
        parents = [by_table[t] for t in order_for_create([s.table for s in parents], ref_columns)]
        children = [by_table[t] for t in order_for_create([s.table for s in children], ref_columns)]

    return [*parents, *children], parent_of


# ------------------------------------------------------------------------ #
# Helpers for diffing one table
# ------------------------------------------------------------------------ #
def default_key(f: Field) -> str | None:
    """A field's default as the single string _field_registry stores and the
    next diff compares against (None = no default). Scalars keep the plain
    str() form older registry rows already hold; a JSON dict/list is stored as
    JSON text, so the same value always compares equal."""
    if f.default is None:
        return None
    if isinstance(f.default, (dict, list)):
        return json.dumps(f.default, sort_keys=True)
    return str(f.default)


def _prev_sql_type(prev: dict) -> str | None:
    """The Postgres type a _field_registry row stands for, or None when it
    cannot be told from the row alone (a REFERENCE to a non-id column, a
    DECIMAL registered before precision/scale were kept)."""
    if prev["type"] == "REFERENCE":
        return None if prev.get("ref_field") else "UUID"
    try:
        return Field(
            id=prev["id"],
            name=prev["name"],
            type=prev["type"],
            length=prev.get("length"),
            precision=prev.get("precision"),
            scale=prev.get("scale"),
        ).sql_type()
    except FieldError:
        return None


#: Type changes that succeed but throw information away (the value or its
#: detail), so the plan calls them out. Every type change is snapshotted to
#: _trash first; these are the ones where the snapshot is likely to matter.
_LOSSY_TYPE_CHANGES = frozenset(
    {
        ("FLOAT", "INT"),
        ("FLOAT", "BIGINT"),
        ("FLOAT", "DECIMAL"),
        ("DECIMAL", "INT"),
        ("DECIMAL", "BIGINT"),
        ("DATETIME", "DATE"),
        ("DATETIME", "TIME"),
        ("INT", "BOOLEAN"),
        ("BIGINT", "INT"),
        ("BIGINT", "BOOLEAN"),
    }
)

#: Postgres type name (as in _prev_sql_type / Field.sql_type, before any "(") ->
#: information_schema.columns.data_type, for the drift check.
_LIVE_DATA_TYPES = {
    "VARCHAR": "character varying",
    "TEXT": "text",
    "BOOLEAN": "boolean",
    "INTEGER": "integer",
    "BIGINT": "bigint",
    "REAL": "real",
    "DOUBLE PRECISION": "double precision",
    "NUMERIC": "numeric",
    "DATE": "date",
    "TIME": "time without time zone",
    "TIMESTAMPTZ": "timestamp with time zone",
    "JSONB": "jsonb",
    "UUID": "uuid",
}


def _drift_warnings(
    schema: TableSchema, mine: dict[str, dict], all_rows: list[dict], live: dict[str, dict]
) -> list[str]:
    """Where the real table differs from what _field_registry says was applied
    — someone changed the database by hand, or a migration was half-reverted.
    Reported, never "fixed" silently: the diff still trusts the registry, so the
    warning is how a human finds out. Checks this plugin's own registry rows
    (a column that is gone, or of another type) and, for a schema, columns
    that exist but that nobody declared."""
    if not live:
        return []
    out: list[str] = []
    for row in mine.values():
        if row["type"] == "TABLE":
            continue
        column = live.get(row["name"])
        if column is None:
            out.append(
                f"drift on '{schema.table}': column '{row['name']}' is in the registry but missing "
                f"from the database (changed outside `arc pgdb migrate`)."
            )
            continue
        expected = _prev_sql_type(row)
        if expected is None:
            continue
        want = _LIVE_DATA_TYPES.get(expected.split("(")[0])
        have = column["data_type"]
        if want and want != have and not (row["type"] == "FLOAT" and have == "real"):
            out.append(
                f"drift on '{schema.table}': column '{row['name']}' is {have} in the database "
                f"but the registry says {row['type']} ({expected})."
            )
    if not schema.is_patch:
        # Another plugin's patched-in columns are in all_rows: theirs, not drift.
        known = {s.name for s in schema.system_fields} | {r["name"] for r in all_rows}
        for name in sorted(set(live) - known):
            out.append(
                f"drift on '{schema.table}': column '{name}' exists in the database but no "
                f"schema or patch declares it."
            )
    return out


async def _refuse_truncation(
    conn: Any,
    schema: TableSchema,
    f: Field,
    prev: dict,
    resolved_sql_type: str,
    prev_sql_type: str | None,
    live: dict[str, dict],
) -> None:
    """A type change to VARCHAR(n) cuts every longer value to n characters,
    silently (an explicit cast truncates). Refuse it at plan time when some
    stored value is longer, before anything runs."""
    new = re.fullmatch(r"VARCHAR\((\d+)\)", resolved_sql_type)
    if new is None:
        return
    new_len = int(new.group(1))
    old = re.fullmatch(r"VARCHAR\((\d+)\)", prev_sql_type or "")
    if old is not None and int(old.group(1)) <= new_len:
        return  # a widening: nothing can be longer than it already was
    column = prev["name"]
    if column not in live:
        return
    longest = await conn.fetchval(f'SELECT max(char_length("{column}"::text)) FROM "{schema.table}"')
    if longest is not None and longest > new_len:
        raise MigrationError(
            f"{'patch' if schema.is_patch else 'schema'} '{schema.source_path.name}' "
            f"(plugin '{schema.plugin}'): field '{f.name}' on '{schema.table}' would become "
            f"{resolved_sql_type}, but a stored value is {longest} characters long — the change "
            f"would cut it short. Shorten or fix those values first, or choose a length of at "
            f"least {longest}."
        )


async def _refuse_orphans(
    conn: Any, schema: TableSchema, f: Field, column: str, ref: "ddl.RefColumn"
) -> None:
    """Making an existing column a REFERENCE adds a foreign key: every value must
    match a row of the target. Checked at plan time when the target already
    exists; a target created in this same plan can't have rows to miss."""
    if not await introspect.table_exists(conn, ref.table):
        return
    orphans = await conn.fetchval(
        f'SELECT count(*) FROM "{schema.table}" t WHERE t."{column}" IS NOT NULL AND NOT EXISTS '
        f'(SELECT 1 FROM "{ref.table}" r WHERE r."{ref.column}"::text = t."{column}"::text)'
    )
    if orphans:
        raise MigrationError(
            f"{'patch' if schema.is_patch else 'schema'} '{schema.source_path.name}' "
            f"(plugin '{schema.plugin}'): field '{f.name}' on '{schema.table}' cannot become a "
            f"REFERENCE to '{ref.table}' — {orphans} stored value(s) match no row in "
            f"{ref.table}.{ref.column}. Fix or clear those values first."
        )


def _column_snapshot_sql(table: str, column: str, *, old_name: str | None = None) -> str:
    """INSERT ... SELECT that copies one column's values into _trash
    (drop_type 'Column'), chunked COLUMN_DROP_CHUNK_SIZE rows per entry, as one
    set-based statement. `old_name` is the column's name in the database right
    now when a rename in the same plan has not run yet at snapshot time."""
    col = old_name or column
    return (
        f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
        f"SELECT '{_q(table)}', 'Column', "
        f"jsonb_agg(jsonb_build_object('_row_id', id, '{_q(col)}', \"{col}\") ORDER BY id), now() "
        f'FROM (SELECT id, "{col}", (row_number() OVER (ORDER BY id) - 1) / {COLUMN_DROP_CHUNK_SIZE} AS chunk '
        f'FROM "{table}") _chunked '
        f"GROUP BY chunk"
    )


async def _diff_indexes(conn: Any, schema: TableSchema, *, new_table: bool) -> list[Op]:
    """Named indexes, compared with the real database and with _index_registry:

      * missing in the database      -> create it (a real, visible op);
      * there, other columns/method  -> drop and create it again;
      * there, on ANOTHER table      -> error: an index key is a database-wide
                                        name, so the second table would silently
                                        get no index;
      * in the registry but no longer declared by this schema/patch -> drop it.

    `new_table=True`: the table is being created in this plan and
    create_table_sql() already creates its indexes, so only the key clash is
    checked here."""
    source: OpSource = "patch" if schema.is_patch else "schema"
    declared = {idx["key"]: idx for idx in schema.indexes}
    ops: list[Op] = []
    for key, idx in declared.items():
        live_idx = await introspect.live_index(conn, key)
        if live_idx is not None and live_idx["table"] != schema.table:
            raise MigrationError(
                f"{source} '{schema.source_path.name}' (plugin '{schema.plugin}'): index key '{key}' "
                f"is already used by table '{live_idx['table']}'. Index keys are database-wide "
                f"names — give this one a name of its own (for example prefix it with "
                f"'{schema.table}_')."
            )
        if new_table:
            continue
        want_columns = list(idx["fields"])
        want_method = ddl.index_method(schema, idx)
        create_sql = ddl.create_index_sql(schema, idx)
        if live_idx is None:
            ops.append(
                Op(
                    kind="create_index",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'{schema.table}: CREATE INDEX "{key}" ({", ".join(want_columns)})',
                    sql=[create_sql],
                )
            )
        elif live_idx["columns"] != want_columns or live_idx["method"] != want_method:
            ops.append(
                Op(
                    kind="create_index",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=(
                        f'{schema.table}: REBUILD INDEX "{key}" (now {", ".join(want_columns)}; '
                        f"was {', '.join(str(c) for c in live_idx['columns'])})"
                    ),
                    sql=[f'DROP INDEX IF EXISTS "{key}"', create_sql],
                )
            )
    if not new_table:
        for row in await introspect.index_registry_rows(conn, schema.table):
            if row["plugin"] == schema.plugin and row["key"] not in declared:
                ops.append(
                    Op(
                        kind="drop_index",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: DROP INDEX "{row["key"]}" (no longer declared)',
                        sql=[f'DROP INDEX IF EXISTS "{row["key"]}"'],
                    )
                )
    return ops


def check_declaration_clashes(schemas: list[TableSchema], patches: list[TableSchema]) -> None:
    """Two files that declare the same field id, the same column name or the
    same index key on one table can't both be right. The diff only compares a
    file with the REGISTRY, so on a first run (nothing registered yet) both are
    "new", both columns are planned, and the registry then records only the
    last one — locking the table's owner out on its next plan. Caught here,
    across everything loaded, before any SQL runs."""
    system_tables = {s.table for s in schemas if s.system}
    ids: dict[tuple[str, str], tuple[TableSchema, Field]] = {}
    names: dict[tuple[str, str], tuple[TableSchema, str, str]] = {}
    index_keys: dict[str, TableSchema] = {}

    def label(decl: TableSchema) -> str:
        kind = "patch" if decl.is_patch else "schema"
        return f"{kind} '{decl.source_path.name}' (plugin '{decl.plugin}')"

    for decl in [*schemas, *patches]:
        if decl.is_patch and decl.table in system_tables:
            continue  # skipped later with a warning, never applied
        for sf in decl.system_fields:
            names[(decl.table, sf.name)] = (decl, sf.id, "a system column")
        for f in decl.fields:
            prior = ids.get((decl.table, f.id))
            if prior is not None and prior[0] is not decl:
                raise MigrationError(
                    f"field id '{f.id}' on table '{decl.table}' is declared by {label(prior[0])} "
                    f"('{prior[1].name}') and by {label(decl)} ('{f.name}'). A field id belongs to "
                    f"one file: give one of them a different id."
                )
            ids[(decl.table, f.id)] = (decl, f)
            if not f.is_column():
                continue
            seen = names.get((decl.table, f.name))
            if seen is not None and seen[0] is not decl:
                holder = (
                    "a system column"
                    if seen[2] == "a system column"
                    else f"declared by {label(seen[0])} ('{seen[1]}')"
                )
                raise MigrationError(
                    f"column '{f.name}' on table '{decl.table}' is declared by {label(decl)} "
                    f"('{f.id}') but is already {holder}. Choose another name."
                )
            names[(decl.table, f.name)] = (decl, f.id, "declared")
        for idx in decl.indexes:
            other = index_keys.get(idx["key"])
            if other is not None and other is not decl:
                raise MigrationError(
                    f"index key '{idx['key']}' is declared by {label(other)} (table '{other.table}') "
                    f"and by {label(decl)} (table '{decl.table}'). Index keys are database-wide "
                    f"names — give one of them a name of its own."
                )
            index_keys[idx["key"]] = decl


# ------------------------------------------------------------------------ #
# Diffing one table (schema OR patch — see module docstring)
# ------------------------------------------------------------------------ #
async def _diff_table(
    conn: Any,
    schema: TableSchema,
    *,
    ref_targets: dict[str, str],
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"],
    parent_table: str | None,
    will_exist: frozenset[str] = frozenset(),
) -> tuple[list[Op], list[str], bool]:
    """Returns (ops, warnings, skipped). `skipped=True` means this schema/
    patch was NOT compared against anything — its target table doesn't
    exist — and callers must NOT include it when upserting _field_registry
    afterward: doing so would record bookkeeping for DDL that never ran.

    `will_exist` — tables a schema IN THIS SAME build_plan call is already
    creating, even though the live database doesn't have them yet (build_plan
    only plans, it never executes its own ops — see module docstring).
    Without this, a patch whose target table is being created for the very
    first time by its own plugin's schema, in the same migrate run, always
    looked like it targeted a nonexistent table and was silently skipped
    with just a warning — needing a SECOND `arc pgdb migrate` before the
    patch's fields ever actually applied. build_plan processes every schema
    before any patch (see its own loop order), so by the time a patch is
    diffed, `will_exist` already reflects every table this pass is creating."""
    source: OpSource = "patch" if schema.is_patch else "schema"
    exists = await introspect.table_exists(conn, schema.table) or schema.table in will_exist

    if not exists:
        if schema.is_patch:
            return (
                [],
                [
                    f"skipped patch '{schema.source_path.name}' (plugin '{schema.plugin}') — "
                    f"table '{schema.table}' does not exist yet."
                ],
                True,
            )
        await _diff_indexes(conn, schema, new_table=True)  # raises on a database-wide key clash
        stmts = ddl.create_table_sql(schema, parent_table=parent_table, ref_columns=ref_columns)
        return (
            [
                Op(
                    kind="create_table",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'CREATE TABLE "{schema.table}" ({len(schema.column_fields())} fields)',
                    sql=stmts,
                    destructive=False,
                )
            ],
            [],
            False,
        )

    all_rows = await introspect.registry_rows(
        conn, schema.table
    )  # every plugin that touches this table
    mine = {row["id"]: row for row in all_rows if row["plugin"] == schema.plugin}
    others = [row for row in all_rows if row["plugin"] != schema.plugin]

    if not schema.is_patch and not mine:
        # A schema claiming to fully own a table that already physically
        # exists, but this plugin has never registered anything for it —
        # either someone else's table, or an unmanaged one. Refuse outright;
        # the table is never touched, let alone dropped.
        if others:
            owners = sorted({r["plugin"] for r in others})
            raise MigrationError(
                f"table '{schema.table}' already exists and is owned by plugin(s) "
                f"{owners} — plugin '{schema.plugin}' cannot declare it as a schema "
                f"of its own. The existing table (and its data) is left untouched; "
                f"rename one side."
            )
        raise MigrationError(
            f"table '{schema.table}' already exists in the database but is not "
            f"tracked in _field_registry (not created by `arc pgdb migrate`) — "
            f"refusing to adopt an unmanaged table automatically. Left untouched."
        )

    ops: list[Op] = []
    current = {f.id: f for f in schema.fields}
    # Column names the registry says are in use on this table right now, whoever
    # owns them: what a new or renamed field must not collide with.
    held_names = {row["name"]: row for row in all_rows}
    live = await introspect.live_columns(conn, schema.table)
    warnings = _drift_warnings(schema, mine, all_rows, live)
    kind_name = "patch" if schema.is_patch else "schema"

    for fid, f in current.items():
        if not f.is_column():
            continue  # TABLE fields never have a column on THIS table — handled as a separate child schema

        if fid not in mine:
            other_owner = next((r["plugin"] for r in others if r["id"] == fid), None)
            if other_owner is not None:
                raise MigrationError(
                    f"{kind_name} '{schema.source_path.name}' (plugin '{schema.plugin}') declares "
                    f"field id '{fid}' ('{f.name}') on table '{schema.table}', but that id "
                    f"already belongs to plugin '{other_owner}' — a plugin may only add new "
                    f"field ids or modify field ids it already owns."
                )
            holder = held_names.get(f.name)
            if holder is not None:
                if holder["plugin"] == schema.plugin and holder["id"] not in current:
                    raise MigrationError(
                        f"{kind_name} '{schema.source_path.name}' (plugin '{schema.plugin}'): field "
                        f"'{f.name}' has a new id '{fid}', but the column '{f.name}' already exists "
                        f"under id '{holder['id']}'. The id is the field's identity: keep id "
                        f"'{holder['id']}' to keep the column and its data, or give the new field a "
                        f"different name to replace it (the old values go to _trash)."
                    )
                raise MigrationError(
                    f"{kind_name} '{schema.source_path.name}' (plugin '{schema.plugin}'): cannot add "
                    f"field '{f.name}' ('{fid}') to '{schema.table}' — a column with that name already "
                    f"exists (field '{holder['id']}', plugin '{holder['plugin']}'). Choose another name."
                )
            col_sql = f'ALTER TABLE "{schema.table}" ADD COLUMN {_column_def(f, schema.table, ref_columns)}'
            # A required column with no default is only safe to add on a
            # table with zero rows — Postgres has nothing to backfill
            # existing rows with, and ADD COLUMN ... NOT NULL (no DEFAULT)
            # fails outright the moment the table has ≥1 row. Every other
            # similarly-risky op here (set_not_null, alter_type, add_unique)
            # is marked destructive=True for exactly this class of risk;
            # add_column was the one exception, always green/"safe" in the
            # plan even when it was this likely to fail at apply time.
            risky = f.required and f.default is None
            ops.append(
                Op(
                    kind="add_column",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=(
                        f'{schema.table}: ADD COLUMN "{f.name}" ({f.type}) NOT NULL, no default — '
                        f"REVIEW: fails if the table already has any rows"
                        if risky
                        else f'{schema.table}: ADD COLUMN "{f.name}" ({f.type})'
                    ),
                    sql=[col_sql],
                    destructive=risky,
                )
            )
            continue

        prev = mine[fid]
        if prev["type"] == "TABLE":
            raise MigrationError(
                f"{kind_name} '{schema.source_path.name}' (plugin '{schema.plugin}'): field "
                f"'{f.name}' ('{fid}') was a TABLE field (a child table, no column) and is now "
                f"{f.type}. That isn't automated — give the new column a new field id."
            )
        if prev["name"] != f.name:
            clash = held_names.get(f.name)
            if clash is not None and clash["id"] != fid:
                raise MigrationError(
                    f"{kind_name} '{schema.source_path.name}' (plugin '{schema.plugin}'): cannot rename "
                    f"'{prev['name']}' to '{f.name}' on '{schema.table}' — the column '{f.name}' still "
                    f"exists (field '{clash['id']}'). Renames that swap or chain names can't run in one "
                    f"step: rename through a temporary name, in two migrations."
                )
            ops.append(
                Op(
                    kind="rename_column",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'{schema.table}: RENAME COLUMN "{prev["name"]}" -> "{f.name}"',
                    sql=[
                        f'ALTER TABLE "{schema.table}" RENAME COLUMN "{prev["name"]}" TO "{f.name}"'
                    ],
                    destructive=False,
                )
            )

        was_reference = prev["type"] == "REFERENCE"
        if f.type == "REFERENCE" and was_reference:
            # Changing what an EXISTING REFERENCE field points at isn't
            # automated — unlike a plain type/length change, this also means
            # the FK constraint itself has to be dropped and recreated
            # against a different column (or a different TABLE entirely),
            # not just the column's type altered. Rather than guess at that
            # sequence (and risk emitting SQL that silently does the wrong
            # thing), this is a hard stop: drop the field and add it again
            # as a new one, going through the existing (already-safe,
            # already-Trash-backed) drop+add path.
            #
            # Checks BOTH target_field (which column on the target) AND the
            # target TABLE itself — only target_field used to be compared
            # here. A schema changing {"target": "Department"} to {"target":
            # "Team"} on an existing field, with target_field unset both
            # times, produced NO diff at all: prev["ref_field"] stayed None
            # -> None (target_field never changed) and f.type stayed
            # "REFERENCE" -> "REFERENCE" (so the type/length check below
            # never fired either) — the live FK constraint kept pointing at
            # the original table forever while the schema file claimed
            # something else, with no error and no warning.
            #
            # A plain column BECOMING a REFERENCE is a different case, handled
            # below: nothing exists yet to repoint, so the foreign key is just
            # added.
            new_target_table = ref_columns[(schema.table, f.name)].table
            ref_field_changed = prev.get("ref_field") != f.target_field
            ref_table_changed = prev.get("ref_table") != new_target_table
            if ref_field_changed or ref_table_changed:
                changed = []
                if ref_table_changed:
                    changed.append(f"target table from {prev.get('ref_table')!r} to {new_target_table!r}")
                if ref_field_changed:
                    changed.append(f"target_field from {prev.get('ref_field')!r} to {f.target_field!r}")
                raise MigrationError(
                    f"{'patch' if schema.is_patch else 'schema'} '{schema.source_path.name}' "
                    f"(plugin '{schema.plugin}'), field '{f.name}': " + " and ".join(changed) + " — "
                    "changing what an existing REFERENCE field points at isn't automated. Drop the "
                    "field and add it again as a new one instead."
                )

        resolved_sql_type = (
            ref_columns[(schema.table, f.name)].sql_type if f.type == "REFERENCE" else f.sql_type()
        )
        # Compared as the real Postgres type each side maps to, not as the
        # declared names: STRING -> EMAIL (both VARCHAR) is not a change,
        # `length: 255` on a STRING that had no length (already VARCHAR(255)) is
        # not a change, and DECIMAL's precision/scale are part of the type.
        # Two REFERENCE fields never differ here (the guard above stopped a
        # retarget).
        prev_sql_type = _prev_sql_type(prev)
        physical_changed = not (was_reference and f.type == "REFERENCE") and (
            prev_sql_type != resolved_sql_type
        )

        if was_reference and f.type != "REFERENCE":
            # REFERENCE -> a plain column: the foreign key has to go too, or the
            # column would keep rejecting values the schema no longer constrains.
            fk_names = await introspect.single_column_constraints(
                conn, schema.table, [prev["name"], f.name], "f"
            )
            ops.append(
                Op(
                    kind="drop_foreign_key",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'{schema.table}: DROP FOREIGN KEY on "{f.name}" (no longer a REFERENCE)',
                    sql=[
                        f'ALTER TABLE "{schema.table}" DROP CONSTRAINT IF EXISTS "{name}"'
                        for name in (fk_names or [f"{schema.table}_{prev['name']}_fkey"])
                    ],
                    destructive=False,
                )
            )

        if physical_changed:
            await _refuse_truncation(conn, schema, f, prev, resolved_sql_type, prev_sql_type, live)
            lossy = (prev["type"], f.type) in _LOSSY_TYPE_CHANGES
            ops.append(
                Op(
                    kind="alter_type",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=(
                        f'{schema.table}: ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} '
                        f"(was {prev['type']}) — REVIEW: "
                        + (
                            "LOSSY (precision, time of day or value detail is lost); "
                            if lossy
                            else "fails if a value cannot be converted; "
                        )
                        + "the old values are saved to _trash first"
                    ),
                    sql=[
                        # The column is copied to _trash BEFORE it is rewritten, the
                        # same safety net a dropped column gets, so a conversion
                        # that loses information can still be undone.
                        _column_snapshot_sql(schema.table, f.name, old_name=prev["name"]),
                        f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} USING "{f.name}"::{resolved_sql_type}',
                    ],
                    destructive=True,
                )
            )
        elif f.type == "FLOAT" and live.get(prev["name"], {}).get("data_type") == "real":
            # FLOAT used to be a 4-byte REAL (about 7 digits); it is now DOUBLE
            # PRECISION. A column created before that still is REAL: widening it
            # loses nothing, so it is done once, without a snapshot.
            ops.append(
                Op(
                    kind="alter_type",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'{schema.table}: ALTER COLUMN "{f.name}" TYPE DOUBLE PRECISION (was REAL — widening, lossless)',
                    sql=[
                        f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" TYPE DOUBLE PRECISION'
                    ],
                    destructive=False,
                )
            )

        if f.type == "REFERENCE" and not was_reference:
            ref = ref_columns[(schema.table, f.name)]
            await _refuse_orphans(conn, schema, f, prev["name"], ref)
            ops.append(
                Op(
                    kind="add_foreign_key",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=(
                        f'{schema.table}: ADD FOREIGN KEY "{f.name}" -> {ref.table}({ref.column}) '
                        f"— REVIEW: fails if a value matches no row there"
                    ),
                    sql=[
                        f'ALTER TABLE "{schema.table}" ADD CONSTRAINT "{schema.table}_{f.name}_fkey" '
                        f'FOREIGN KEY ("{f.name}") REFERENCES "{ref.table}"("{ref.column}") '
                        f"ON DELETE RESTRICT"
                    ],
                    destructive=True,
                )
            )

        if bool(prev["reqd"]) != f.required:
            if f.required:
                ops.append(
                    Op(
                        kind="set_not_null",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: SET NOT NULL on "{f.name}" — REVIEW: fails if existing rows have NULL',
                        sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" SET NOT NULL'],
                        destructive=True,
                    )
                )
            else:
                ops.append(
                    Op(
                        kind="drop_not_null",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: DROP NOT NULL on "{f.name}"',
                        sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" DROP NOT NULL'],
                        destructive=False,
                    )
                )

        if bool(prev["unique"]) != f.unique:
            if f.unique:
                constraint = f"{schema.table}_{f.name}_key"
                ops.append(
                    Op(
                        kind="add_unique",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: ADD UNIQUE ("{f.name}") — REVIEW: fails if duplicates already exist',
                        sql=[
                            f'ALTER TABLE "{schema.table}" ADD CONSTRAINT "{constraint}" UNIQUE ("{f.name}")'
                        ],
                        destructive=True,
                    )
                )
            else:
                # The constraint keeps the name it was CREATED under
                # ("{table}_{column}_key" for the column's name back then): a
                # RENAME COLUMN renames the column only, never the constraint,
                # in this pass or in an earlier migrate. Guessing the name from
                # the current (or even the previous) column name therefore
                # missed it, silently leaving uniqueness in force while the
                # registry said "not unique". It is found by COLUMN instead —
                # the column is called prev["name"] now (a rename in this same
                # plan has not run yet) or f.name (renamed by an earlier run).
                found = await introspect.single_column_constraints(
                    conn, schema.table, [prev["name"], f.name], "u"
                )
                ops.append(
                    Op(
                        kind="drop_unique",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: DROP UNIQUE ("{f.name}")',
                        sql=[
                            f'ALTER TABLE "{schema.table}" DROP CONSTRAINT IF EXISTS "{name}"'
                            for name in (found or [f"{schema.table}_{prev['name']}_key"])
                        ],
                        destructive=False,
                    )
                )

        new_default = default_key(f)
        if not f.primary_key and new_default != prev.get("default"):
            if f.default is None:
                ops.append(
                    Op(
                        kind="drop_default",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: DROP DEFAULT on "{f.name}" (existing rows keep their values)',
                        sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" DROP DEFAULT'],
                        destructive=False,
                    )
                )
            else:
                ops.append(
                    Op(
                        kind="set_default",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=(
                            f'{schema.table}: SET DEFAULT on "{f.name}" to {ddl.default_sql(f)} '
                            f"(new rows only; existing rows keep their values)"
                        ),
                        sql=[
                            f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" '
                            f"SET DEFAULT {ddl.default_sql(f)}"
                        ],
                        destructive=False,
                    )
                )

    for fid, prev in mine.items():
        if fid in current:
            continue
        if prev["type"] == "TABLE":
            continue  # a child-table field never had a column here, so there is nothing to drop
        # A field is only "missing, therefore dropped" if the file being
        # diffed right now is the same KIND of file that declared it in the
        # first place (found by testing this against a real DB: adding a
        # brand new field to the SCHEMA file, after a PATCH had already
        # added an unrelated field to the same table, made the schema's own
        # diff think it needed to DROP the patch's field — the schema was
        # never responsible for declaring it, so its absence from the
        # schema's own current fields means nothing). Symmetric with the
        # patch side directly below: a schema file is the declaration for
        # schema-sourced fields, a table's ONE patch file is the
        # declaration for patch-sourced fields — neither is a full listing
        # of the other's fields, so neither's mere silence about the
        # other's field is ever a removal signal.
        expected_source = "patch" if schema.is_patch else "schema"
        if prev.get("source", "schema") != expected_source:
            continue
        ops.append(_drop_column_op(schema.table, schema.plugin, prev, source=source))

    ops += await _diff_indexes(conn, schema, new_table=False)
    return ops, warnings, False


async def _diff_unique_together(
    conn: Any, schema: TableSchema, *, table_exists: bool, bootstrapped: bool
) -> list[Op]:
    """Diffs schema.unique_together — the WHOLE declared list for this
    table, not per-field — against _unique_together's registry rows,
    ownership-scoped to schema.plugin exactly like _diff_table's own field
    diffing (so one plugin's re-migrate never proposes dropping another
    plugin's group on a table they both touch). Schema-only by
    construction: unique_together is always `[]` on a patch
    (schema_spec.PatchFileSpec has no such key), so this is only ever
    called from the schemas loop in build_plan, never for patches.

    `table_exists=False` is a deliberate, total no-op — a brand-new table
    gets every declared group rendered INLINE by ddl.create_table_sql (a
    CREATE TABLE statement can just list them as table constraints, no
    ALTER needed); diffing here too would propose adding the exact same
    constraint a second time.

    `bootstrapped=False` is also a total no-op, for a different reason:
    build_plan only *queues* the bootstrap op (BOOTSTRAP_STRUCTURAL_SQL,
    which creates _unique_together) — it never executes its own plan, so
    on a database bootstrapped before _unique_together existed, the table
    genuinely isn't there yet during THIS build_plan call. Querying it
    would be a hard UndefinedTableError on every already-existing table,
    not just ones with a declared unique_together group. Skipping here is
    safe: the very next build_plan call, after apply_plan has run the
    queued bootstrap op, sees bootstrapped=True and diffs for real."""
    if not table_exists or not bootstrapped:
        return []

    rows = await introspect.unique_together_rows(conn, schema.table)
    mine = {r["key"]: r for r in rows if r["plugin"] == schema.plugin}
    declared = {ut["key"]: ut for ut in schema.unique_together}

    ops: list[Op] = []
    for key, ut in declared.items():
        if key in mine:
            if sorted(mine[key]["fields"]) != sorted(ut["fields"]):
                # Same "id/key is the stable identity, not the fields
                # behind it" posture as a REFERENCE field's target change
                # (§ _diff_table above) — changing the columns of an
                # existing composite constraint isn't automated, since
                # that's really a different constraint wearing the same
                # name. Hard stop, before any SQL runs.
                raise MigrationError(
                    f"schema '{schema.source_path.name}' (plugin '{schema.plugin}'): "
                    f"unique_together '{key}' changed its fields from "
                    f"{mine[key]['fields']} to {ut['fields']} — changing an existing "
                    f"composite constraint's columns isn't automated. Remove '{key}' "
                    f"and declare a new group under a different key instead."
                )
            continue
        exists = await introspect.constraint_exists(conn, schema.table, key)
        if exists:
            # Already there physically even though the registry doesn't
            # know it yet (e.g. applied by an interrupted earlier run) —
            # defensive idempotency, same reasoning create_table_sql's own
            # `IF NOT EXISTS` already leans on elsewhere in this module.
            continue
        cols = ", ".join(ut["fields"])
        ops.append(
            Op(
                kind="add_unique_together",
                table=schema.table,
                plugin=schema.plugin,
                source="schema",
                description=(
                    f'{schema.table}: ADD CONSTRAINT "{key}" UNIQUE ({cols}) — '
                    f"REVIEW: fails if duplicate combinations already exist"
                ),
                sql=[ddl.unique_together_sql(schema.table, ut)],
                destructive=True,
            )
        )

    for key, row in mine.items():
        if key in declared:
            continue
        ops.append(
            Op(
                kind="drop_unique_together",
                table=schema.table,
                plugin=schema.plugin,
                source="schema",
                description=f'{schema.table}: DROP CONSTRAINT "{key}"',
                sql=[ddl.drop_unique_together_sql(schema.table, key)],
                destructive=False,
            )
        )
    return ops


def _column_def(
    f: Field, owner_table: str, ref_columns: dict[tuple[str, str], "ddl.RefColumn"]
) -> str:
    from .ddl import _user_column_sql  # noqa: PLC0415 - internal helper, not part of ddl's public surface

    return _user_column_sql(f, owner_table=owner_table, ref_columns=ref_columns)


#: Every existing row's value for a dropped column is snapshotted into
#: _trash in groups of this many rows per _trash entry, not one _trash row
#: per table row — see _drop_column_op's own docstring for why an
#: unchunked design (one row, or one row-per-table-row) doesn't work here.
COLUMN_DROP_CHUNK_SIZE = 10_000


def _drop_column_op(
    table: str, plugin: str, prev: dict, *, source: OpSource, forget_registry: bool = False
) -> Op:
    """Every existing row's value for this column is snapshotted into
    _trash BEFORE the column is actually dropped — chunked into groups of
    COLUMN_DROP_CHUNK_SIZE rows per _trash entry (one jsonb ARRAY of
    {"_row_id": ..., "<col>": ...} objects per chunk), single set-based
    INSERT, not a per-row Python loop.

    Two unchunked designs were considered and rejected first: one _trash
    row per table row (the original design — floods _trash with millions
    of one-value rows on a large table) and one _trash row for the WHOLE
    column (a single jsonb array covering every row). The second one hits
    a real, hard Postgres ceiling most people don't know about: jsonb's
    on-disk format encodes each element's offset/length in 28 bits
    (JENTRY_OFFLENMASK), capping any single jsonb VALUE at 268,435,455
    bytes (~256MB) — independent of, and far tighter than, the general
    1GB TOAST limit other varlena types get. A wide dropped column (a
    TEXT/JSON field, not a short string) on a large table blows past that
    and fails the migration outright, mid-DROP-COLUMN. Chunking bounds the
    worst case per _trash row to COLUMN_DROP_CHUNK_SIZE regardless of
    table size or column width, while still cutting a million-row drop
    from a million _trash rows down to ~100.

    Reached both when a schema removes a field it owns and when a patch
    removes a field it previously added — identical treatment either way."""
    col = prev["name"]
    snapshot_sql = _column_snapshot_sql(table, col)
    drop_sql = f'ALTER TABLE "{table}" DROP COLUMN "{col}"'
    sql = [snapshot_sql, drop_sql]
    if forget_registry:
        # No schema/patch is declared for these registry rows any more, so the
        # normal "rewrite this plugin's rows" step never reaches them.
        sql.append(
            f"DELETE FROM _field_registry WHERE \"table\" = '{_q(table)}' AND id = '{_q(prev['id'])}'"
        )
    return Op(
        kind="drop_column",
        table=table,
        plugin=plugin,
        source=source,
        description=(
            f'{table}: DROP COLUMN "{col}" — every existing value snapshotted to _trash first '
            f"(chunked, {COLUMN_DROP_CHUNK_SIZE} rows/entry)"
        ),
        sql=sql,
        destructive=True,
    )


# ------------------------------------------------------------------------ #
# Public entry points
# ------------------------------------------------------------------------ #
async def build_plan(
    conn: Any,
    schemas: list[TableSchema],
    patches: list[TableSchema] | None = None,
    *,
    only_table: str | None = None,
    allow_drops: bool = False,
) -> MigrationPlan:
    """Diff everything declared against the database. `allow_drops=False` (the
    default) never plans the removal of anything that is merely no longer
    DECLARED — a table, or a patch's column: "not declared" is also what a
    disabled plugin, another git branch or a renamed file looks like, and
    none of those means "delete the data". Such things come back as warnings;
    `allow_drops=True` (`arc pgdb migrate --allow-drops`) plans the drops,
    each with its rows copied to _trash first."""
    patches = patches or []
    bootstrapped = await introspect.bootstrap_applied(conn)
    plan = MigrationPlan()

    # Shared trigger functions are CREATE OR REPLACE and re-applied on every
    # migrate (not just the first) — see ddl.py's module docstring on why.
    # Ordered BEFORE the structural bootstrap: _trash (and every _audit_*
    # table) declares DEFAULT arc_uuid_generate_v7(), so the function must
    # exist by the time those CREATE TABLE statements run. Creating the
    # functions first is safe on a fresh database — plpgsql doesn't resolve
    # gen_random_bytes()/table references until execution, so nothing here
    # needs pgcrypto or _trash to exist yet.
    plan.ops.append(
        Op(
            kind="create_table",
            table="_bootstrap",
            plugin="pgdb",
            description="ensure shared trigger functions (arc_uuid_generate_v7, "
            "arc_set_updated_at, arc_soft_delete_to_trash) are current",
            sql=list(ddl.BOOTSTRAP_FUNCTIONS_SQL),
            destructive=False,
            maintenance=True,
        )
    )
    if not bootstrapped:
        plan.ops.append(
            Op(
                kind="create_table",
                table="_bootstrap",
                plugin="pgdb",
                description="bootstrap: pgcrypto extension, _field_registry, _trash, "
                "_patch_history, _routines_history",
                sql=list(ddl.BOOTSTRAP_STRUCTURAL_SQL),
                destructive=False,
            )
        )

    _resolve_child_owners(schemas)  # validates ownership; raises MigrationError on violation
    refuse_patches_on_own_tables(schemas, patches)  # a plugin never patches a table it owns
    check_declaration_clashes(schemas, patches)  # one id/name/index key = one file
    system_tables = {s.table for s in schemas if s.system}
    ref_targets = resolve_ref_targets([*schemas, *patches])
    ref_columns = resolve_ref_columns(
        [*schemas, *patches], ref_targets
    )  # validates target_field ownership/uniqueness
    ordered, parent_of = _order_and_link(schemas, ref_targets, ref_columns)

    targets = [*ordered, *patches]
    if only_table:
        targets = [s for s in targets if s.table == only_table]
        if not targets:
            raise MigrationError(f"no declared schema or patch produces table '{only_table}'.")

    applied: list[
        TableSchema
    ] = []  # schemas/patches actually diffed — NOT skipped ones (see _diff_table)
    seen_plugins: set[str] = set()
    # Tables a schema in THIS pass creates for the first time — build_plan
    # never executes its own ops (pure planning), so the live database
    # still doesn't have them by the time patches are diffed below, even
    # though they're about to exist once apply_plan runs. Without this, a
    # patch targeting a table its OWN plugin's schema creates in this same
    # migrate run always looked like it targeted a nonexistent table and
    # was silently skipped — see _diff_table's `will_exist` docstring.
    will_exist: set[str] = set()
    for schema in ordered:
        if only_table and schema not in targets:
            continue
        if schema.audit and schema.plugin not in seen_plugins:
            plan.ops.append(
                Op(
                    kind="create_table",
                    table=f"_audit_{schema.plugin}",
                    plugin=schema.plugin,
                    description=f'ensure audit table "_audit_{schema.plugin}" exists',
                    sql=ddl.audit_table_sql(schema.plugin),
                    destructive=False,
                    maintenance=True,
                )
            )
            # Only mark a plugin "seen" once its audit table has actually
            # been ensured — adding it unconditionally here (for EVERY
            # schema, audited or not) meant a plugin whose first-processed
            # schema has "audit": false poisoned this set before a LATER
            # audited schema from the same plugin was ever reached, and the
            # _audit_<plugin> table/function silently never got created.
            # Never triggered before authn: every previously-audited plugin
            # (example_hr) happened to have its audited schema ordered
            # first, which masked this exact ordering dependency.
            seen_plugins.add(schema.plugin)

        table_ops, warns, skipped = await _diff_table(
            conn,
            schema,
            ref_targets=ref_targets,
            ref_columns=ref_columns,
            parent_table=parent_of.get(schema.table),
        )
        plan.ops.extend(table_ops)
        plan.warnings.extend(warns)
        if not skipped:
            applied.append(schema)
            # A brand-new table already got every declared unique_together
            # group rendered INLINE by _diff_table's own create_table_sql
            # call above — checking table_exists here (never introspect'd
            # before this point; build_plan never executes its own ops, so
            # the live database is unchanged regardless of what table_ops
            # just computed) is what lets _diff_unique_together tell "new
            # table, already handled" from "existing table, needs a real
            # ALTER" apart, with no extra parameter threaded through
            # _diff_table's own signature.
            table_exists_now = await introspect.table_exists(conn, schema.table)
            plan.ops.extend(
                await _diff_unique_together(
                    conn, schema, table_exists=table_exists_now, bootstrapped=bootstrapped
                )
            )
        if any(op.kind == "create_table" and op.table == schema.table for op in table_ops):
            will_exist.add(schema.table)

        if schema.audit:
            plan.ops.append(
                Op(
                    kind="ensure_index",
                    table=schema.table,
                    plugin=schema.plugin,
                    description=f"{schema.table}: attach audit trigger",
                    sql=ddl.audit_attach_sql(schema.table, schema.plugin),
                    destructive=False,
                    maintenance=True,
                )
            )
        elif not skipped and table_exists_now:
            # `audit` turned off: the trigger from when it was on would keep
            # writing audit rows. Dropped here; the audit table and the history
            # already in it stay.
            trigger = f"arc_audit_{schema.plugin}"
            if await introspect.trigger_exists(conn, schema.table, trigger):
                plan.ops.append(
                    Op(
                        kind="drop_trigger",
                        table=schema.table,
                        plugin=schema.plugin,
                        description=(
                            f"{schema.table}: DROP audit trigger (audit is now off; "
                            f"the audit history already recorded is kept)"
                        ),
                        sql=[f'DROP TRIGGER IF EXISTS {trigger} ON "{schema.table}"'],
                        destructive=False,
                    )
                )

    for patch in patches:
        if only_table and patch not in targets:
            continue
        if patch.table in system_tables:
            # Patches never apply to system tables — a "system": true table
            # self-declares its ENTIRE structure itself (pgdb.model);
            # letting some other plugin patch fields onto it would mean its
            # shape is no longer fully self-declared. Same "skip, don't
            # error" treatment as a patch aimed at a table that doesn't
            # exist yet — the target may simply not have been designed to
            # accept patches, not a mistake worth halting a migrate over.
            plan.warnings.append(
                f"skipped patch '{patch.source_path.name}' (plugin '{patch.plugin}') — "
                f"table '{patch.table}' is a system table; patches cannot target system tables."
            )
            continue
        patch_ops, warns, skipped = await _diff_table(
            conn,
            patch,
            ref_targets=ref_targets,
            ref_columns=ref_columns,
            parent_table=None,
            will_exist=frozenset(will_exist),
        )
        plan.ops.extend(patch_ops)
        plan.warnings.extend(warns)
        if not skipped:
            applied.append(patch)

    if not only_table:
        drop_ops, drop_warnings = await _dropped_table_ops(
            conn, ordered, allow_drops=allow_drops
        )
        plan.ops.extend(drop_ops)
        plan.warnings.extend(drop_warnings)
    orphan_ops, orphan_warnings = await _orphaned_patch_ops(
        conn, ordered, patches, only_table=only_table, allow_drops=allow_drops
    )
    plan.ops.extend(orphan_ops)
    plan.warnings.extend(orphan_warnings)

    # Only schemas/patches that were ACTUALLY diffed — a skipped patch (its
    # target table doesn't exist yet) must never reach registry_upsert_sql,
    # or _field_registry would record fields for DDL that never ran.
    plan.schemas = applied
    plan.ref_targets = ref_targets
    plan.ref_columns = ref_columns
    return plan


async def apply_plan(conn: Any, plan: MigrationPlan, *, reference: str) -> None:
    """Executes every op's SQL in order, overwrites _field_registry to
    match, and records _patch_history — all in one transaction so a
    failure partway through never leaves bookkeeping out of sync with the
    actual tables."""
    async with conn.transaction():
        # A migration is not an application query. The pool's statement timeout
        # (30 s by default) and idle-in-transaction timeout would cancel a long
        # CREATE INDEX, type rewrite or _trash snapshot on a big table and roll
        # the whole migration back, so they are lifted for THIS transaction only
        # (SET LOCAL). A DDL that cannot get its lock gives up instead: left
        # waiting, it would also queue every query that arrives behind it.
        await conn.execute("SET LOCAL statement_timeout = 0")
        await conn.execute("SET LOCAL idle_in_transaction_session_timeout = 0")
        await conn.execute(f"SET LOCAL lock_timeout = '{MIGRATION_LOCK_TIMEOUT}'")
        for op in plan.ops:
            for stmt in op.sql:
                await conn.execute(stmt)
        for stmt in registry_upsert_sql(plan.schemas, plan.ref_targets):
            await conn.execute(stmt)
        for stmt in unique_together_upsert_sql(plan.schemas):
            await conn.execute(stmt)
        for stmt in index_registry_upsert_sql(plan.schemas):
            await conn.execute(stmt)
        for stmt in patch_history_sql(plan.ops, reference):
            await conn.execute(stmt)


async def _orphaned_patch_ops(
    conn: Any,
    declared: list[TableSchema],
    patches: list[TableSchema],
    *,
    only_table: str | None,
    allow_drops: bool,
) -> tuple[list[Op], list[str]]:
    """Columns a patch added whose patch FILE is gone (deleted, or its plugin
    disabled). Removing one FIELD from a patch that stays is a normal diff; a
    missing patch is only "not declared", which is not the same as "remove".
    With `allow_drops` each such column is dropped (values copied to _trash
    first, registry row forgotten); otherwise it is reported and kept."""
    declared_patch_keys = {(p.table, p.plugin) for p in patches}
    declared_tables = {s.table for s in declared}
    by_owner: dict[tuple[str, str], list[dict]] = {}
    for row in await introspect.patch_registry_rows(conn):
        key = (row["table"], row["plugin"])
        if key in declared_patch_keys or row["table"] not in declared_tables:
            continue  # still declared, or the whole table is handled as a table drop
        if only_table and row["table"] != only_table:
            continue
        by_owner.setdefault(key, []).append(row)
    ops: list[Op] = []
    warnings: list[str] = []
    for (table, plugin), rows in sorted(by_owner.items()):
        columns = sorted(r["name"] for r in rows if r["type"] != "TABLE")
        if not columns:
            continue
        if not allow_drops:
            warnings.append(
                f"no patch from plugin '{plugin}' is declared for table '{table}' any more, but "
                f"it added the column(s) {columns}. Kept as they are (a missing patch file is "
                f"also a disabled plugin). Run `arc pgdb migrate --allow-drops` to drop them; "
                f"their values are saved to _trash first."
            )
            continue
        ops.extend(
            _drop_column_op(table, plugin, row, source="patch", forget_registry=True)
            for row in rows
            if row["type"] != "TABLE"
        )
    return ops, warnings


async def _dropped_table_ops(
    conn: Any, declared: list[TableSchema], *, allow_drops: bool = False
) -> tuple[list[Op], list[str]]:
    """Tables in the registry that no loaded schema declares. "Not declared"
    also describes a disabled plugin, a plugin that failed to load, another git
    branch and a renamed schema file — so without `allow_drops` nothing is
    dropped and each table is reported once. With it, the table's rows are
    copied to _trash and the table dropped."""
    declared_names = {s.table for s in declared}
    if not await introspect.bootstrap_applied(conn):
        return [], []
    registry = await conn.fetch('select "table", id, plugin from _field_registry')
    ids_by_table: dict[str, set[str]] = {}
    owners_by_table: dict[str, set[str]] = {}
    for row in registry:
        ids_by_table.setdefault(row["table"], set()).add(row["id"])
        owners_by_table.setdefault(row["table"], set()).add(row["plugin"])
    new_tables = {
        s.table: {f.id for f in s.fields}
        for s in declared
        if s.table not in ids_by_table and not s.is_patch
    }
    ops: list[Op] = []
    warnings: list[str] = []
    for table in sorted(ids_by_table):
        if table in declared_names:
            continue
        # A table can have more than one owning plugin (a schema plus
        # patches from others) — attribute the drop to the alphabetically
        # first owner. Cosmetic only: which plugin's generated .sql file
        # mentions it, not which plugin's data gets dropped (all of it does,
        # snapshotted to _trash first, regardless of owner).
        plugin = sorted(owners_by_table[table])[0]
        renamed_to = next((n for n, ids in new_tables.items() if ids == ids_by_table[table]), None)
        hint = (
            f" It has exactly the fields of the new table '{renamed_to}': this looks like a "
            f"RENAMED schema file. A rename is not automated — dropping '{table}' would leave "
            f"'{renamed_to}' empty."
            if renamed_to
            else ""
        )
        if not allow_drops:
            warnings.append(
                f"table '{table}' (plugin '{plugin}') is in the database but no loaded schema "
                f"declares it — NOT dropped. A disabled or failing plugin, another git branch "
                f"or a renamed file all look like this.{hint} To drop it on purpose, run "
                f"`arc pgdb migrate --allow-drops` (its rows are saved to _trash first)."
            )
            continue
        snapshot_sql = (
            f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
            f"SELECT '{_q(table)}', 'Table', to_jsonb(t), now() FROM \"{table}\" t"
        )
        ops.append(
            Op(
                kind="drop_table",
                table=table,
                plugin=plugin,
                description=f'DROP TABLE "{table}" — every row snapshotted to _trash first' + hint,
                sql=[
                    snapshot_sql,
                    # CASCADE, not a plain DROP: when more than one
                    # undeclared table is being dropped in the same plan
                    # and they reference each other (e.g. a plugin with
                    # several interrelated tables gets fully removed),
                    # `rows` above has no guaranteed dependency order —
                    # whichever side happens to be a referenced table's
                    # target can be visited first, and a plain DROP TABLE
                    # then fails with DependentObjectsStillExistError.
                    # CASCADE only drops what actually depends on THIS
                    # table (its own FK constraints, indexes, etc.) — it
                    # does not touch any other table's ROWS — so a
                    # still-being-dropped referencing table just loses the
                    # now-moot constraint and gets its own snapshot+drop
                    # normally when its own turn comes.
                    f'DROP TABLE IF EXISTS "{table}" CASCADE',
                    f"DELETE FROM _field_registry WHERE \"table\" = '{_q(table)}'",
                    f"DELETE FROM _unique_together WHERE \"table\" = '{_q(table)}'",
                    f"DELETE FROM _index_registry WHERE \"table\" = '{_q(table)}'",
                ],
                destructive=True,
            )
        )
    return ops, warnings


def registry_upsert_sql(schemas: list[TableSchema], ref_targets: dict[str, str]) -> list[str]:
    """After applying a plan, _field_registry is overwritten to match the
    schemas/patches just applied — this IS the "previous state" the next
    diff runs against. Scoped by (table, plugin) — NOT just table — so
    upserting one plugin's fields never wipes out another plugin's rows on
    a table with more than one owner (a schema plus someone's patch).
    `ref_table` is stored as the resolved, slugified physical table name
    (never the raw schema-file target string) — the soft-delete cascade
    trigger executes it directly as an identifier. `ref_field` is stored
    exactly as declared (None for the default 'id' target) — this is what
    the NEXT diff compares target_field against (see _diff_table's
    target_field-change guard); it doesn't need cross-schema resolution to
    write, only to validate, and that validation already happened whenever
    this plan's ref_columns was built. `source` ('schema' or 'patch')
    records which kind of file declared each field — see _diff_table's
    patch drop-detection, which must never treat a SCHEMA-owned field as
    removed just because one particular patch file doesn't redeclare it.

    The DELETE is scoped by (table, plugin) — NOT by (table, plugin, kind)
    — and issued only ONCE per (table, plugin) pair even though `schemas`
    can contain BOTH a schema and one or more patches from the same plugin
    on the same table (patching your own schema-created table is legal).
    Issuing a separate DELETE per entry would have the patch's own delete
    wipe out the schema's just-inserted rows (and vice versa) — same table,
    same plugin, so the same scope — before either finished writing its
    own fields, silently corrupting the registry despite every column
    still being physically present."""
    stmts = []
    seen_table_plugin: set[tuple[str, str]] = set()
    for schema in schemas:
        key = (schema.table, schema.plugin)
        if key not in seen_table_plugin:
            stmts.append(
                f"DELETE FROM _field_registry WHERE \"table\" = '{_q(schema.table)}' "
                f"AND plugin = '{_q(schema.plugin)}'"
            )
            seen_table_plugin.add(key)
        source = "patch" if schema.is_patch else "schema"
        for f in schema.fields:
            length = f.length if f.length is not None else "NULL"
            # precision/scale — DECIMAL only, NULL otherwise (mirrors length's
            # own "NULL unless this type actually uses it" convention above).
            # Stored so the next diff can actually SEE a precision/scale
            # change (see _diff_table's decimal_changed) — before these two
            # columns existed, a DECIMAL(10,2) -> DECIMAL(12,4) edit produced
            # no diff op at all, silently leaving the live column on its
            # original precision/scale forever.
            precision = f.precision if f.type == "DECIMAL" and f.precision is not None else "NULL"
            scale = f.scale if f.type == "DECIMAL" and f.scale is not None else "NULL"
            default = f"'{_q(default_key(f))}'" if f.default is not None else "NULL"
            ref_table = (
                f"'{_q(ref_targets[f.target])}'" if f.type in ("REFERENCE", "TABLE") else "NULL"
            )
            ref_field = (
                f"'{_q(f.target_field)}'"
                if f.type == "REFERENCE" and f.target_field is not None
                else "NULL"
            )
            # ON CONFLICT, not a plain INSERT: a patch legitimately
            # redeclaring a field id ITS OWN plugin's schema also declares
            # (pgdb.schema()'s own merge already treats this as "a
            # legitimate rename/retype... supersedes the schema's version")
            # used to crash outright with a duplicate-key violation on
            # ("table", id) — the DELETE above runs once per (table,
            # plugin), not once per entry, so a schema's own INSERT for an
            # id and a same-plugin patch's INSERT for that SAME id both
            # landed in the same transaction with nothing to stop the
            # second from colliding with the first. Patches are always
            # processed (and so always appear later in `schemas`, i.e.
            # `plan.schemas`) after their own plugin's schema — see
            # build_plan's loop order — so "last one in this list wins"
            # naturally means the patch's version wins, matching schema()'s
            # own override semantics exactly rather than a coincidence.
            stmts.append(
                "INSERT INTO _field_registry "
                '(id, name, "table", type, length, precision, scale, reqd, "unique", "default", '
                "ref_table, ref_field, source, plugin) "
                f"VALUES ('{_q(f.id)}', '{_q(f.name)}', '{_q(schema.table)}', '{_q(f.type)}', {length}, "
                f"{precision}, {scale}, {f.required}, {f.unique}, {default}, {ref_table}, {ref_field}, "
                f"'{source}', '{_q(schema.plugin)}') "
                'ON CONFLICT ("table", id) DO UPDATE SET '
                "name = EXCLUDED.name, type = EXCLUDED.type, length = EXCLUDED.length, "
                "precision = EXCLUDED.precision, scale = EXCLUDED.scale, reqd = EXCLUDED.reqd, "
                '"unique" = EXCLUDED."unique", "default" = EXCLUDED."default", '
                "ref_table = EXCLUDED.ref_table, ref_field = EXCLUDED.ref_field, "
                "source = EXCLUDED.source, plugin = EXCLUDED.plugin, updated_at = now()"
            )
    return stmts


def unique_together_upsert_sql(schemas: list[TableSchema]) -> list[str]:
    """After applying a plan, _unique_together is overwritten to match the
    groups just applied — this IS the "previous state" _diff_unique_together
    diffs the next declared list against, the same role registry_upsert_sql
    plays for _field_registry. Same (table, plugin)-scoped, once-per-pair
    DELETE-then-INSERT shape, and for the identical reason: a patch can
    never declare unique_together (always `[]`), but IS unconditionally
    present in `schemas` — deduping by (table, plugin) means a patch never
    re-runs (and never wipes) the delete+insert its own schema already did
    for the same pair, matching build_plan's loop order (schema before its
    patches)."""
    stmts = []
    seen_table_plugin: set[tuple[str, str]] = set()
    for schema in schemas:
        key = (schema.table, schema.plugin)
        if key in seen_table_plugin:
            continue
        seen_table_plugin.add(key)
        stmts.append(
            f"DELETE FROM _unique_together WHERE \"table\" = '{_q(schema.table)}' "
            f"AND plugin = '{_q(schema.plugin)}'"
        )
        for ut in schema.unique_together:
            fields_sql = "ARRAY[" + ", ".join(f"'{_q(f)}'" for f in ut["fields"]) + "]"
            stmts.append(
                'INSERT INTO _unique_together ("table", "key", fields, plugin) '
                f"VALUES ('{_q(schema.table)}', '{_q(ut['key'])}', {fields_sql}, '{_q(schema.plugin)}') "
                'ON CONFLICT ("table", "key") DO UPDATE SET '
                "fields = EXCLUDED.fields, plugin = EXCLUDED.plugin, updated_at = now()"
            )
    return stmts


def index_registry_upsert_sql(schemas: list[TableSchema]) -> list[str]:
    """After applying a plan, _index_registry is rewritten to the named indexes
    each schema/patch declares — what the next diff compares against to notice a
    removed index. Same (table, plugin)-scoped DELETE-then-INSERT shape as the
    field and unique_together registries; an index key is database-wide, so the
    INSERT also takes over a row that moved to another table."""
    stmts = []
    seen: set[tuple[str, str]] = set()
    for schema in schemas:
        key = (schema.table, schema.plugin)
        if key not in seen:
            seen.add(key)
            stmts.append(
                f"DELETE FROM _index_registry WHERE \"table\" = '{_q(schema.table)}' "
                f"AND plugin = '{_q(schema.plugin)}'"
            )
        for idx in schema.indexes:
            fields_sql = "ARRAY[" + ", ".join(f"'{_q(f)}'" for f in idx["fields"]) + "]"
            stmts.append(
                'INSERT INTO _index_registry ("key", "table", fields, plugin) '
                f"VALUES ('{_q(idx['key'])}', '{_q(schema.table)}', {fields_sql}, '{_q(schema.plugin)}') "
                'ON CONFLICT ("key") DO UPDATE SET "table" = EXCLUDED."table", '
                "fields = EXCLUDED.fields, plugin = EXCLUDED.plugin, updated_at = now()"
            )
    return stmts


def patch_history_sql(ops: list[Op], reference: str) -> list[str]:
    """One row per distinct (plugin, table, kind) actually touched by
    `ops`. Only pgdb's own infra ops are excluded — the "_bootstrap"
    sentinel and "_audit_*" table creation. A business-declared
    `"system": true` table (authn's `_users`, relay's `_job_log`, ...)
    legitimately starts with "_" and IS part of the history — the earlier
    blanket `startswith("_")` skip silently excluded every one of them,
    contradicting this table's own "every applied change" contract."""
    seen: set[tuple[str, str, str]] = set()
    stmts = []
    for op in ops:
        if op.table == "_bootstrap" or op.table.startswith("_audit_"):
            continue
        key = (op.plugin, op.table, op.source)
        if key in seen:
            continue
        seen.add(key)
        stmts.append(
            'INSERT INTO _patch_history (plugin, "table", reference, kind) '
            f"VALUES ('{_q(op.plugin)}', '{_q(op.table)}', '{_q(reference)}', '{op.source}') "
            'ON CONFLICT (plugin, "table", kind, reference) DO NOTHING'
        )
    return stmts


def migration_reference() -> str:
    """A single timestamp shared by every plugin's generated migration file
    for one `arc pgdb migrate` run — used both as the filename and as
    _patch_history.reference."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


# --------------------------------------------------------------------- #
# Lifecycle events — emitted by EVERY path that applies a migration (the
# `arc pgdb migrate` CLI and admin's "Apply Now"), through the two helpers
# below, so the two can never drift apart. Subscribers use plain
# `arc.events.on(...)` from a plugin's events.py.
#
#   pgdb.premigration   after the operator confirmed and the migration lock
#                       is held, immediately BEFORE anything is applied.
#                       The database still has the OLD shape.
#   pgdb.postmigration  after everything applied and the lock was released.
#                       Never fired for a migration that failed or was
#                       aborted. The database has the NEW shape.
#
# Both fire only when there is real work (a real schema change, or pending
# routines on the CLI) — a no-op or maintenance-only run is silent, or every
# subscriber would wake on every deploy. A handler's failure is logged and
# reported in the summary but never fails or vetoes the migration (see
# arc.events.emit), so these are for reacting (enqueue a lineup job, clear a
# cache, notify), not for gating.
#
# Payload (schema_version 1 — fields are only ever ADDED, never changed or
# removed):
#   schema_version  int
#   source          "cli" | "admin"
#   reference       the migration reference (also _patch_history.reference)
#   plugin / table  the requested scope, or None for a system-wide run
#   changes         [{plugin, table, kind, source, destructive, description}]
#                   — the plan's real (non-maintenance) ops
#   routines        [{plugin, name}] — the routines about to run (pre) /
#                   that ran (post)
# --------------------------------------------------------------------- #
PRE_MIGRATION_EVENT = "pgdb.premigration"
POST_MIGRATION_EVENT = "pgdb.postmigration"
MIGRATION_EVENT_SCHEMA_VERSION = 1


def migration_event_payload(
    plan: MigrationPlan,
    *,
    source: Literal["cli", "admin"],
    reference: str,
    plugin: str | None = None,
    table: str | None = None,
    routines: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MIGRATION_EVENT_SCHEMA_VERSION,
        "source": source,
        "reference": reference,
        "plugin": plugin,
        "table": table,
        "changes": [
            {
                "plugin": op.plugin,
                "table": op.table,
                "kind": op.kind,
                "source": op.source,
                "destructive": op.destructive,
                "description": op.description,
            }
            for op in plan.real_ops()
        ],
        "routines": [{"plugin": r.plugin, "name": r.name} for r in routines or []],
    }


async def emit_migration_event(name: str, payload: dict[str, Any]) -> dict[str, str]:
    """arc.events.emit, plus a log line per failed handler. Returns the
    {handler: "ok" | "error: ..."} map so a CLI can print a summary."""
    results = await arc.events.emit(name, **payload)
    for handler, outcome in results.items():
        if outcome != "ok":
            _logger.warning("%s handler %s: %s", name, handler, outcome)
    return results


def write_migration_file(
    plugin_dir: Path, plugin: str, plan: MigrationPlan, reference: str
) -> Path:
    migrations_dir = plugin_dir / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    path = migrations_dir / f"{reference}_migration.sql"
    lines = [f"-- generated by `arc pgdb migrate` for plugin '{plugin}'", ""]
    for op in plan.ops:
        if op.plugin != plugin:
            continue
        marker = "-- [DESTRUCTIVE]" if op.destructive else "--"
        lines.append(f"{marker} ({op.source}) {op.description}")
        lines.extend(f"{stmt};" for stmt in op.sql)
        lines.append("")
    path.write_text("\n".join(lines))
    return path
