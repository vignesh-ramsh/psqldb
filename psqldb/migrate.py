"""
psqldb.migrate
-------------------
Orchestrates everything `arc psqldb plan` / `arc psqldb migrate` do:
collect declared schemas AND patches -> diff each against `_field_registry`
(the DB's memory of "the schema as of the last successful migration", see
psqldb.introspect) -> render a plan -> optionally apply it, updating
`_field_registry` to match and recording the change in `_patch_history`.

Two declaration sources, one diff engine:
  * schemas/<Table>.json — a plugin fully owns and creates this table.
  * patches/<Table>.json — a plugin adds fields to / modifies fields it
    already owns on a table that already exists (possibly owned by a
    DIFFERENT plugin's schema). Never creates a table (skipped with a
    warning if the target doesn't exist yet, or if the target is a
    `"system": true` table — those self-declare their entire structure,
    psqldb.model, so no other plugin's patch ever applies to one); can drop
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

Deliberate simplifications for this first cut (not oversights — narrower
than the full design on purpose, safe to widen later without a schema
change):
  * child tables may not themselves declare a TABLE field (no nested
    children yet) — validated, hard error if violated.
  * named composite indexes are only ever added (CREATE INDEX IF NOT
    EXISTS); one removed from a schema/patch file is left in place rather
    than auto-dropped. Costs disk/write performance, never data — an
    acceptable v1 gap, unlike column/table drops which go through the
    diff+trash path.
  * a widened/narrowed column type is always treated as destructive review,
    not classified more finely (widening a VARCHAR is obviously safe;
    stop-and-review-everything is the simpler, safer default for now).
"""

from __future__ import annotations

import datetime as dt
import heapq
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Literal

from . import ddl, introspect
from .fields import Field
from .model import SchemaError, TableSchema, load_schemas_dir

OpKind = Literal[
    "create_table", "drop_table", "add_column", "drop_column",
    "rename_column", "alter_type", "set_not_null", "drop_not_null",
    "add_unique", "drop_unique", "ensure_index",
]
OpSource = Literal["schema", "patch"]


@dataclass
class Op:
    kind: OpKind
    table: str
    plugin: str
    description: str          # human-readable, shown in `plan`/`migrate`
    sql: list[str]            # statement(s) that implement this op
    destructive: bool = False
    source: OpSource = "schema"


@dataclass
class MigrationPlan:
    ops: list[Op] = dc_field(default_factory=list)
    schemas: list[TableSchema] = dc_field(default_factory=list)   # schemas + patches this plan covers
    ref_targets: dict[str, str] = dc_field(default_factory=dict)  # schema-stem -> physical table name
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"] = dc_field(default_factory=dict)  # (table, field name) -> resolved target
    warnings: list[str] = dc_field(default_factory=list)          # e.g. "skipped patch for missing table"

    def destructive_ops(self) -> list[Op]:
        return [op for op in self.ops if op.destructive]

    def by_table(self) -> dict[str, list[Op]]:
        out: dict[str, list[Op]] = {}
        for op in self.ops:
            out.setdefault(op.table, []).append(op)
        return out

    def is_empty(self) -> bool:
        return not self.ops


class MigrationError(RuntimeError):
    pass


# ------------------------------------------------------------------------ #
# Ordering + relational resolution — schemas only; patches never create a
# table or own a child, so they never participate in this.
# ------------------------------------------------------------------------ #
def _resolve_child_owners(schemas: list[TableSchema]) -> dict[str, TableSchema]:
    """Every `child: true` schema must be targeted by exactly one TABLE
    field somewhere. Returns {child_schema_stem: owning TableSchema}."""
    by_stem = {s.source_path.stem: s for s in schemas}
    owners: dict[str, list[str]] = {}
    for s in schemas:
        for f in s.child_fields():
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


def resolve_ref_targets(schemas: list[TableSchema]) -> dict[str, str]:
    """Maps each REFERENCE/TABLE field's raw `target` (a schema-file stem,
    e.g. "Employee") to the physical, slugified table it resolves to (e.g.
    "employee"). Used both for DDL rendering and for insert()-time FK
    existence checks (psqldb.validation). Callers pass schemas + patches
    together so a patch's own REFERENCE fields resolve against the same
    known-table set schemas do."""
    by_stem = {s.source_path.stem: s for s in schemas}
    ref_targets: dict[str, str] = {}
    for s in schemas:
        for f in s.fields:
            if f.type in ("REFERENCE", "TABLE") and f.target:
                target_schema = by_stem.get(f.target)
                if target_schema is None:
                    raise MigrationError(
                        f"schema '{s.table}', field '{f.name}': target '{f.target}' "
                        f"does not match any known schema file name."
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
    a patch can add a new REFERENCE field with its own target_field."""
    by_stem = {s.source_path.stem: s for s in schemas}
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"] = {}
    for s in schemas:
        for f in s.fields:
            if f.type != "REFERENCE" or not f.target:
                continue
            target_table = ref_targets[f.target]
            if f.target_field is None:
                ref_columns[(s.table, f.name)] = ddl.RefColumn(table=target_table, column="id", sql_type="UUID")
                continue
            target_schema = by_stem[f.target]
            target_field = next((tf for tf in target_schema.fields if tf.name == f.target_field), None)
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
                    f"\"unique\": true — Postgres can only create a foreign key "
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
    """Orders a set of tables so `arc psqldb clear -p <plugin>` (which
    clears every table a plugin owns in one pass) respects REFERENCE
    dependencies among them: a table that references another via a real
    FK (ON DELETE RESTRICT, §3.9's REFERENCE type) must be cleared FIRST —
    clear's soft-delete-to-trash trigger issues a real physical DELETE the
    moment a row's _state hits 99, and that DELETE fails outright with a
    live ForeignKeyViolationError if some OTHER row (in this same clear
    batch) still references it. Found by running `arc psqldb clear -p
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
            f"(`arc psqldb clear -t <table>`), breaking the cycle by hand first."
        )
    return order


def _order_and_link(
    schemas: list[TableSchema], ref_targets: dict[str, str]
) -> tuple[list[TableSchema], dict[str, str]]:
    """Returns (ordered schemas — parents before children, parent_of map
    "child table" -> "parent table")."""
    by_stem = {s.source_path.stem: s for s in schemas}
    parent_of: dict[str, str] = {}
    for s in schemas:
        for f in s.child_fields():
            child_schema = by_stem[f.target]
            parent_of[child_schema.table] = s.table

    parents = [s for s in schemas if not s.child]
    children = [s for s in schemas if s.child]
    return [*parents, *children], parent_of


# ------------------------------------------------------------------------ #
# Diffing one table (schema OR patch — see module docstring)
# ------------------------------------------------------------------------ #
async def _diff_table(
    conn: Any, schema: TableSchema, *, ref_targets: dict[str, str],
    ref_columns: dict[tuple[str, str], "ddl.RefColumn"], parent_table: str | None,
) -> tuple[list[Op], list[str], bool]:
    """Returns (ops, warnings, skipped). `skipped=True` means this schema/
    patch was NOT compared against anything — its target table doesn't
    exist — and callers must NOT include it when upserting _field_registry
    afterward: doing so would record bookkeeping for DDL that never ran."""
    source: OpSource = "patch" if schema.is_patch else "schema"
    exists = await introspect.table_exists(conn, schema.table)

    if not exists:
        if schema.is_patch:
            return [], [
                f"skipped patch '{schema.source_path.name}' (plugin '{schema.plugin}') — "
                f"table '{schema.table}' does not exist yet."
            ], True
        stmts = ddl.create_table_sql(schema, parent_table=parent_table, ref_columns=ref_columns)
        return [Op(
            kind="create_table", table=schema.table, plugin=schema.plugin, source=source,
            description=f'CREATE TABLE "{schema.table}" ({len(schema.column_fields())} fields)',
            sql=stmts, destructive=False,
        )], [], False

    all_rows = await introspect.registry_rows(conn, schema.table)  # every plugin that touches this table
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
            f"tracked in _field_registry (not created by `arc psqldb migrate`) — "
            f"refusing to adopt an unmanaged table automatically. Left untouched."
        )

    ops: list[Op] = []
    current = {f.id: f for f in schema.fields}

    for fid, f in current.items():
        if not f.is_column():
            continue  # TABLE fields never have a column on THIS table — handled as a separate child schema

        if fid not in mine:
            other_owner = next((r["plugin"] for r in others if r["id"] == fid), None)
            if other_owner is not None:
                kind = "patch" if schema.is_patch else "schema"
                raise MigrationError(
                    f"{kind} '{schema.source_path.name}' (plugin '{schema.plugin}') declares "
                    f"field id '{fid}' ('{f.name}') on table '{schema.table}', but that id "
                    f"already belongs to plugin '{other_owner}' — a plugin may only add new "
                    f"field ids or modify field ids it already owns."
                )
            col_sql = f'ALTER TABLE "{schema.table}" ADD COLUMN {_column_def(f, schema.table, ref_columns)}'
            ops.append(Op(
                kind="add_column", table=schema.table, plugin=schema.plugin, source=source,
                description=f'{schema.table}: ADD COLUMN "{f.name}" ({f.type})',
                sql=[col_sql], destructive=False,
            ))
            continue

        prev = mine[fid]
        if prev["name"] != f.name:
            ops.append(Op(
                kind="rename_column", table=schema.table, plugin=schema.plugin, source=source,
                description=f'{schema.table}: RENAME COLUMN "{prev["name"]}" -> "{f.name}"',
                sql=[f'ALTER TABLE "{schema.table}" RENAME COLUMN "{prev["name"]}" TO "{f.name}"'],
                destructive=False,
            ))

        if f.type == "REFERENCE" and prev.get("ref_field") != f.target_field:
            # Changing what an EXISTING REFERENCE field points at isn't
            # automated — unlike a plain type/length change, this also means
            # the FK constraint itself has to be dropped and recreated
            # against a different column, not just the column's type altered.
            # Rather than guess at that sequence (and risk emitting SQL that
            # silently does the wrong thing), this is a hard stop: drop the
            # field and add it again as a new one, going through the
            # existing (already-safe, already-Trash-backed) drop+add path.
            raise MigrationError(
                f"{'patch' if schema.is_patch else 'schema'} '{schema.source_path.name}' "
                f"(plugin '{schema.plugin}'), field '{f.name}': target_field changed "
                f"from {prev.get('ref_field')!r} to {f.target_field!r} — changing what "
                f"an existing REFERENCE field points at isn't automated. Drop the field "
                f"and add it again as a new one instead."
            )

        resolved_sql_type = ref_columns[(schema.table, f.name)].sql_type if f.type == "REFERENCE" else f.sql_type()
        type_changed = prev["type"] != f.type
        length_changed = prev.get("length") != f.length and f.type in ("STRING", "SELECT", "EMAIL", "PHONE")
        if type_changed or length_changed:
            ops.append(Op(
                kind="alter_type", table=schema.table, plugin=schema.plugin, source=source,
                description=f'{schema.table}: ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} (was {prev["type"]}) — REVIEW: may fail or truncate data',
                sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} USING "{f.name}"::{resolved_sql_type}'],
                destructive=True,
            ))

        if bool(prev["reqd"]) != f.required:
            if f.required:
                ops.append(Op(
                    kind="set_not_null", table=schema.table, plugin=schema.plugin, source=source,
                    description=f'{schema.table}: SET NOT NULL on "{f.name}" — REVIEW: fails if existing rows have NULL',
                    sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" SET NOT NULL'],
                    destructive=True,
                ))
            else:
                ops.append(Op(
                    kind="drop_not_null", table=schema.table, plugin=schema.plugin, source=source,
                    description=f'{schema.table}: DROP NOT NULL on "{f.name}"',
                    sql=[f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" DROP NOT NULL'],
                    destructive=False,
                ))

        if bool(prev["unique"]) != f.unique:
            constraint = f"{schema.table}_{f.name}_key"
            if f.unique:
                ops.append(Op(
                    kind="add_unique", table=schema.table, plugin=schema.plugin, source=source,
                    description=f'{schema.table}: ADD UNIQUE ("{f.name}") — REVIEW: fails if duplicates already exist',
                    sql=[f'ALTER TABLE "{schema.table}" ADD CONSTRAINT "{constraint}" UNIQUE ("{f.name}")'],
                    destructive=True,
                ))
            else:
                ops.append(Op(
                    kind="drop_unique", table=schema.table, plugin=schema.plugin, source=source,
                    description=f'{schema.table}: DROP UNIQUE ("{f.name}")',
                    sql=[f'ALTER TABLE "{schema.table}" DROP CONSTRAINT IF EXISTS "{constraint}"'],
                    destructive=False,
                ))

    for fid, prev in mine.items():
        if fid in current:
            continue
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
        ops.append(_drop_column_op(schema, prev, source=source))

    ops += [
        Op(kind="ensure_index", table=schema.table, plugin=schema.plugin, source=source,
           description=f'{schema.table}: ensure index "{idx["key"]}"', sql=[stmt], destructive=False)
        for idx, stmt in zip(schema.indexes, ddl.index_sql(schema))
    ]
    return ops, [], False


def _column_def(f: Field, owner_table: str, ref_columns: dict[tuple[str, str], "ddl.RefColumn"]) -> str:
    from .ddl import _user_column_sql  # noqa: PLC0415 - internal helper, not part of ddl's public surface
    return _user_column_sql(f, owner_table=owner_table, ref_columns=ref_columns)


def _drop_column_op(schema: TableSchema, prev: dict, *, source: OpSource) -> Op:
    """Every existing row's value for this column is snapshotted into
    _trash (one _trash row per table row, single set-based INSERT — not a
    per-row Python loop) BEFORE the column is actually dropped. Reached
    both when a schema removes a field it owns and when a patch removes a
    field it previously added — identical treatment either way."""
    col = prev["name"]
    snapshot_sql = (
        f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
        f'SELECT \'{schema.table}\', \'Column\', '
        f'jsonb_build_object(\'_row_id\', id, \'{col}\', "{col}"), now() '
        f'FROM "{schema.table}"'
    )
    drop_sql = f'ALTER TABLE "{schema.table}" DROP COLUMN "{col}"'
    return Op(
        kind="drop_column", table=schema.table, plugin=schema.plugin, source=source,
        description=f'{schema.table}: DROP COLUMN "{col}" — every existing value snapshotted to _trash first',
        sql=[snapshot_sql, drop_sql], destructive=True,
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
) -> MigrationPlan:
    patches = patches or []
    bootstrapped = await introspect.bootstrap_applied(conn)
    plan = MigrationPlan()

    if not bootstrapped:
        plan.ops.append(Op(
            kind="create_table", table="_bootstrap", plugin="psqldb",
            description="bootstrap: pgcrypto extension, _field_registry, _trash, _patch_history",
            sql=list(ddl.BOOTSTRAP_STRUCTURAL_SQL), destructive=False,
        ))
    # Shared trigger functions are CREATE OR REPLACE and re-applied on every
    # migrate (not just the first) — see ddl.py's module docstring on why.
    plan.ops.append(Op(
        kind="create_table", table="_bootstrap", plugin="psqldb",
        description="ensure shared trigger functions (arc_set_updated_at, "
                    "arc_soft_delete_to_trash) are current",
        sql=list(ddl.BOOTSTRAP_FUNCTIONS_SQL), destructive=False,
    ))

    _resolve_child_owners(schemas)  # validates ownership; raises MigrationError on violation
    system_tables = {s.table for s in schemas if s.system}
    ref_targets = resolve_ref_targets([*schemas, *patches])
    ref_columns = resolve_ref_columns([*schemas, *patches], ref_targets)  # validates target_field ownership/uniqueness
    ordered, parent_of = _order_and_link(schemas, ref_targets)

    targets = [*ordered, *patches]
    if only_table:
        targets = [s for s in targets if s.table == only_table]
        if not targets:
            raise MigrationError(f"no declared schema or patch produces table '{only_table}'.")

    applied: list[TableSchema] = []  # schemas/patches actually diffed — NOT skipped ones (see _diff_table)
    seen_plugins: set[str] = set()
    for schema in ordered:
        if only_table and schema not in targets:
            continue
        if schema.audit and schema.plugin not in seen_plugins:
            plan.ops.append(Op(
                kind="create_table", table=f"_audit_{schema.plugin}", plugin=schema.plugin,
                description=f'ensure audit table "_audit_{schema.plugin}" exists',
                sql=ddl.audit_table_sql(schema.plugin), destructive=False,
            ))
        seen_plugins.add(schema.plugin)

        table_ops, warns, skipped = await _diff_table(
            conn, schema, ref_targets=ref_targets, ref_columns=ref_columns, parent_table=parent_of.get(schema.table)
        )
        plan.ops.extend(table_ops)
        plan.warnings.extend(warns)
        if not skipped:
            applied.append(schema)

        if schema.audit:
            plan.ops.append(Op(
                kind="ensure_index", table=schema.table, plugin=schema.plugin,
                description=f'{schema.table}: attach audit trigger',
                sql=ddl.audit_attach_sql(schema.table, schema.plugin), destructive=False,
            ))

    for patch in patches:
        if only_table and patch not in targets:
            continue
        if patch.table in system_tables:
            # Patches never apply to system tables — a "system": true table
            # self-declares its ENTIRE structure itself (psqldb.model);
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
            conn, patch, ref_targets=ref_targets, ref_columns=ref_columns, parent_table=None
        )
        plan.ops.extend(patch_ops)
        plan.warnings.extend(warns)
        if not skipped:
            applied.append(patch)

    if not only_table:
        plan.ops.extend(await _dropped_table_ops(conn, ordered))

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
        for op in plan.ops:
            for stmt in op.sql:
                await conn.execute(stmt)
        for stmt in registry_upsert_sql(plan.schemas, plan.ref_targets):
            await conn.execute(stmt)
        for stmt in patch_history_sql(plan.ops, reference):
            await conn.execute(stmt)


async def _dropped_table_ops(conn: Any, declared: list[TableSchema]) -> list[Op]:
    declared_names = {s.table for s in declared}
    rows = await conn.fetch('select distinct "table" from _field_registry') \
        if await introspect.bootstrap_applied(conn) else []
    ops = []
    for row in rows:
        table = row["table"]
        if table in declared_names:
            continue
        # A table can have more than one owning plugin (a schema plus
        # patches from others) — attribute the drop to the alphabetically
        # first owner. Cosmetic only: which plugin's generated .sql file
        # mentions it, not which plugin's data gets dropped (all of it does,
        # snapshotted to _trash first, regardless of owner).
        owners = await conn.fetch('select distinct plugin from _field_registry where "table" = $1', table)
        plugin = sorted(r["plugin"] for r in owners)[0] if owners else "unknown"
        snapshot_sql = (
            f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
            f'SELECT \'{table}\', \'Table\', to_jsonb(t), now() FROM "{table}" t'
        )
        ops.append(Op(
            kind="drop_table", table=table, plugin=plugin,
            description=f'DROP TABLE "{table}" — every row snapshotted to _trash first',
            sql=[snapshot_sql, f'DROP TABLE IF EXISTS "{table}"', f'DELETE FROM _field_registry WHERE "table" = \'{table}\''],
            destructive=True,
        ))
    return ops


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
                f"DELETE FROM _field_registry WHERE \"table\" = '{schema.table}' "
                f"AND plugin = '{schema.plugin}'"
            )
            seen_table_plugin.add(key)
        source = "patch" if schema.is_patch else "schema"
        for f in schema.fields:
            length = f.length if f.length is not None else "NULL"
            default = f"'{f.default}'" if f.default is not None else "NULL"
            ref_table = f"'{ref_targets[f.target]}'" if f.type in ("REFERENCE", "TABLE") else "NULL"
            ref_field = f"'{f.target_field}'" if f.type == "REFERENCE" and f.target_field is not None else "NULL"
            stmts.append(
                "INSERT INTO _field_registry "
                '(id, name, "table", type, length, reqd, "unique", "default", ref_table, ref_field, source, plugin) '
                f"VALUES ('{f.id}', '{f.name}', '{schema.table}', '{f.type}', {length}, "
                f"{f.required}, {f.unique}, {default}, {ref_table}, {ref_field}, '{source}', '{schema.plugin}')"
            )
    return stmts


def patch_history_sql(ops: list[Op], reference: str) -> list[str]:
    """One row per distinct (plugin, table, kind) actually touched by
    `ops`. Bootstrap/infra ops (table names starting with "_", e.g. the
    "_bootstrap" sentinel or an audit-table's own creation) are excluded —
    this is a business-schema history, not an internal log."""
    seen: set[tuple[str, str, str]] = set()
    stmts = []
    for op in ops:
        if op.table.startswith("_"):
            continue
        key = (op.plugin, op.table, op.source)
        if key in seen:
            continue
        seen.add(key)
        stmts.append(
            'INSERT INTO _patch_history (plugin, "table", reference, kind) '
            f"VALUES ('{op.plugin}', '{op.table}', '{reference}', '{op.source}') "
            'ON CONFLICT (plugin, "table", kind, reference) DO NOTHING'
        )
    return stmts


def migration_reference() -> str:
    """A single timestamp shared by every plugin's generated migration file
    for one `arc psqldb migrate` run — used both as the filename and as
    _patch_history.reference."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def write_migration_file(plugin_dir: Path, plugin: str, plan: MigrationPlan, reference: str) -> Path:
    migrations_dir = plugin_dir / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    path = migrations_dir / f"{reference}_migration.sql"
    lines = [f"-- generated by `arc psqldb migrate` for plugin '{plugin}'", ""]
    for op in plan.ops:
        if op.plugin != plugin:
            continue
        marker = "-- [DESTRUCTIVE]" if op.destructive else "--"
        lines.append(f"{marker} ({op.source}) {op.description}")
        lines.extend(f"{stmt};" for stmt in op.sql)
        lines.append("")
    path.write_text("\n".join(lines))
    return path
