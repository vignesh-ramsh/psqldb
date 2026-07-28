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

import asyncio
import contextlib
import datetime as dt
import heapq
import logging
import zlib
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Literal

from . import ddl, introspect
from .fields import Field
from .model import SchemaError, TableSchema, load_schemas_dir

_logger = logging.getLogger("psqldb")

# One fixed key shared by every `arc psqldb migrate` run against every
# database — a stable (deterministic, not Python's salted hash()) 32-bit
# value derived from a fixed string, well within Postgres's signed-bigint
# advisory-lock key range. Not a real secret, not table/schema-specific:
# the whole point is that ANY two migrate runs against the SAME database
# contend for the SAME lock, regardless of which plugin/table triggered
# them.
MIGRATE_LOCK_KEY = zlib.crc32(b"arc_psqldb_migrate")

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

    def is_empty(self) -> bool:
        return not self.ops


class MigrationError(RuntimeError):
    pass


@contextlib.asynccontextmanager
async def migration_lock(conn: Any, *, poll_interval: float = 1.0):
    """Serializes concurrent `arc psqldb migrate` runs against the same
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
                "another `arc psqldb migrate` is already running against this "
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
# recomputing psqldb's own cached ref_targets()).
_warned_table_name_targets: set[str] = set()


def resolve_ref_targets(schemas: list[TableSchema]) -> dict[str, str]:
    """Maps each REFERENCE/TABLE field's raw `target` to the physical,
    slugified table it resolves to (e.g. "Employee" -> "employee"). Used
    both for DDL rendering and for insert()-time FK existence checks
    (psqldb.validation). Callers pass schemas + patches together so a
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
    a patch can add a new REFERENCE field with its own target_field."""
    by_stem = {s.source_path.stem: s for s in schemas}
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
            target_schema = by_stem[f.target]
            target_field = next(
                (tf for tf in target_schema.fields if tf.name == f.target_field), None
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


def order_for_create(
    tables: list[str], ref_columns: dict[tuple[str, str], "ddl.RefColumn"]
) -> list[str]:
    """Orders a set of tables so a fresh `arc psqldb migrate` creates a
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
    order (psqldb.model.load_schemas_dir), which has no reason to match
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
    with just a warning — needing a SECOND `arc psqldb migrate` before the
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
        if prev["name"] != f.name:
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

        if f.type == "REFERENCE":
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
        type_changed = prev["type"] != f.type
        length_changed = prev.get("length") != f.length and f.type in (
            "STRING",
            "SELECT",
            "EMAIL",
            "PHONE",
        )
        # DECIMAL's own two dimensions — precision/scale — used to be
        # invisible to this diff entirely: _field_registry had no columns to
        # remember them in, so DECIMAL(10,2) -> DECIMAL(12,4) produced zero
        # ops, silently leaving the live column on its original shape
        # forever. `scale` defaults to 0 in both the schema (Field.sql_type)
        # and here, so an unset scale on either side compares equal to an
        # explicit 0 rather than a spurious mismatch.
        decimal_changed = f.type == "DECIMAL" and (
            prev.get("precision") != f.precision or (prev.get("scale") or 0) != (f.scale or 0)
        )
        if type_changed or length_changed or decimal_changed:
            ops.append(
                Op(
                    kind="alter_type",
                    table=schema.table,
                    plugin=schema.plugin,
                    source=source,
                    description=f'{schema.table}: ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} (was {prev["type"]}) — REVIEW: may fail or truncate data',
                    sql=[
                        f'ALTER TABLE "{schema.table}" ALTER COLUMN "{f.name}" TYPE {resolved_sql_type} USING "{f.name}"::{resolved_sql_type}'
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
            constraint = f"{schema.table}_{f.name}_key"
            if f.unique:
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
                ops.append(
                    Op(
                        kind="drop_unique",
                        table=schema.table,
                        plugin=schema.plugin,
                        source=source,
                        description=f'{schema.table}: DROP UNIQUE ("{f.name}")',
                        sql=[
                            f'ALTER TABLE "{schema.table}" DROP CONSTRAINT IF EXISTS "{constraint}"'
                        ],
                        destructive=False,
                    )
                )

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
        Op(
            kind="ensure_index",
            table=schema.table,
            plugin=schema.plugin,
            source=source,
            description=f'{schema.table}: ensure index "{idx["key"]}"',
            sql=[stmt],
            destructive=False,
        )
        for idx, stmt in zip(schema.indexes, ddl.index_sql(schema))
    ]
    return ops, [], False


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


def _drop_column_op(schema: TableSchema, prev: dict, *, source: OpSource) -> Op:
    """Every existing row's value for this column is snapshotted into
    _trash (one _trash row per table row, single set-based INSERT — not a
    per-row Python loop) BEFORE the column is actually dropped. Reached
    both when a schema removes a field it owns and when a patch removes a
    field it previously added — identical treatment either way."""
    col = prev["name"]
    snapshot_sql = (
        f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
        f"SELECT '{_q(schema.table)}', 'Column', "
        f"jsonb_build_object('_row_id', id, '{_q(col)}', \"{col}\"), now() "
        f'FROM "{schema.table}"'
    )
    drop_sql = f'ALTER TABLE "{schema.table}" DROP COLUMN "{col}"'
    return Op(
        kind="drop_column",
        table=schema.table,
        plugin=schema.plugin,
        source=source,
        description=f'{schema.table}: DROP COLUMN "{col}" — every existing value snapshotted to _trash first',
        sql=[snapshot_sql, drop_sql],
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
) -> MigrationPlan:
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
            plugin="psqldb",
            description="ensure shared trigger functions (arc_uuid_generate_v7, "
            "arc_set_updated_at, arc_soft_delete_to_trash) are current",
            sql=list(ddl.BOOTSTRAP_FUNCTIONS_SQL),
            destructive=False,
        )
    )
    if not bootstrapped:
        plan.ops.append(
            Op(
                kind="create_table",
                table="_bootstrap",
                plugin="psqldb",
                description="bootstrap: pgcrypto extension, _field_registry, _trash, _patch_history",
                sql=list(ddl.BOOTSTRAP_STRUCTURAL_SQL),
                destructive=False,
            )
        )

    _resolve_child_owners(schemas)  # validates ownership; raises MigrationError on violation
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
                )
            )

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
        for stmt in unique_together_upsert_sql(plan.schemas):
            await conn.execute(stmt)
        for stmt in patch_history_sql(plan.ops, reference):
            await conn.execute(stmt)


async def _dropped_table_ops(conn: Any, declared: list[TableSchema]) -> list[Op]:
    declared_names = {s.table for s in declared}
    rows = (
        await conn.fetch('select distinct "table" from _field_registry')
        if await introspect.bootstrap_applied(conn)
        else []
    )
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
        owners = await conn.fetch(
            'select distinct plugin from _field_registry where "table" = $1', table
        )
        plugin = sorted(r["plugin"] for r in owners)[0] if owners else "unknown"
        snapshot_sql = (
            f'INSERT INTO _trash ("table", drop_type, snapshot, deleted_at) '
            f"SELECT '{_q(table)}', 'Table', to_jsonb(t), now() FROM \"{table}\" t"
        )
        ops.append(
            Op(
                kind="drop_table",
                table=table,
                plugin=plugin,
                description=f'DROP TABLE "{table}" — every row snapshotted to _trash first',
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
                ],
                destructive=True,
            )
        )
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
            default = f"'{_q(f.default)}'" if f.default is not None else "NULL"
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
            # (psqldb.schema()'s own merge already treats this as "a
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


def patch_history_sql(ops: list[Op], reference: str) -> list[str]:
    """One row per distinct (plugin, table, kind) actually touched by
    `ops`. Only psqldb's own infra ops are excluded — the "_bootstrap"
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
    for one `arc psqldb migrate` run — used both as the filename and as
    _patch_history.reference."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def write_migration_file(
    plugin_dir: Path, plugin: str, plan: MigrationPlan, reference: str
) -> Path:
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
