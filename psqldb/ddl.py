"""
psqldb.ddl
-------------------
Renders SQL text from a psqldb.model.TableSchema. Nothing in this module
executes anything — it only produces strings, so `arc psqldb plan` can show
exactly what would run without a DB connection touching real data.

Three kinds of output:
  1. bootstrap_sql()        — once per project: pgcrypto, the two shared
                               trigger functions, and the three system
                               tables (_trash, _field_registry, _audit_*
                               are created per-plugin as they're needed).
  2. create_table_sql(...)  — a brand new table (system fields included).
  3. AlterOp render methods — one ALTER TABLE statement per differ.Op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .fields import Field
from .model import CHILD_SYSTEM_FIELDS, NORMAL_SYSTEM_FIELDS, TableSchema


@dataclass(frozen=True)
class RefColumn:
    """What one REFERENCE field's target actually resolves to, once every
    plugin's schemas are known — computed in psqldb.migrate.resolve_ref_columns
    (needs the full cross-plugin schema set, so it can't live on Field itself;
    see Field.target_field). `column`/`sql_type` are "id"/"UUID" for the
    default case (no target_field declared) — same shape either way so
    callers here never need two code paths for "default" vs "custom" target."""
    table: str       # physical target table
    column: str       # physical target column ("id", or a declared unique field's name)
    sql_type: str     # that column's own SQL type — the REFERENCING column must match it

# ------------------------------------------------------------------------ #
# System fields don't go through Field.sql_type() (they use sentinel types
# not in fields.CANONICAL_TYPES) — rendered explicitly here instead.
# ------------------------------------------------------------------------ #
def _system_column_sql(f: Field, *, parent_table: str | None) -> str:
    if f.id == "_id":
        # arc_uuid_generate_v7(), not Postgres's own gen_random_uuid() (v4) —
        # see BOOTSTRAP_FUNCTIONS_SQL below for why and how.
        return '"id" UUID PRIMARY KEY DEFAULT arc_uuid_generate_v7()'
    if f.id == "_parent":
        assert parent_table is not None, "child table system fields require a resolved parent"
        return f'"parent" UUID NOT NULL REFERENCES "{parent_table}"(id) ON DELETE CASCADE'
    if f.id == "_idx":
        return '"idx" INTEGER NOT NULL DEFAULT 0'
    if f.id in ("_created_at", "_updated_at"):
        return f'"{f.name}" TIMESTAMPTZ NOT NULL DEFAULT now()'
    if f.id in ("_created_by", "_updated_by"):
        return f'"{f.name}" UUID'  # nullable, no FK — psqldb doesn't know a "users" table exists (§3.3)
    if f.id == "_state":
        return '"_state" INTEGER NOT NULL DEFAULT 0'
    raise AssertionError(f"unhandled system field id {f.id!r}")


def _user_column_sql(f: Field, *, owner_table: str, ref_columns: dict[tuple[str, str], RefColumn]) -> str:
    ref = ref_columns.get((owner_table, f.name)) if f.type == "REFERENCE" else None
    # `ref` is resolved (and, for a non-default target_field, validated as
    # pointing at a real "unique": true column) by psqldb.migrate before this
    # ever runs — this function only renders, it never decides (module
    # docstring). f.sql_type() itself would silently be wrong here (always
    # "UUID" for REFERENCE — see Field.target_field's docstring), so a
    # resolved `ref` always wins when this is a REFERENCE column.
    sql_type = ref.sql_type if ref is not None else f.sql_type()
    parts = [f'"{f.name}"', sql_type]
    if f.required:
        parts.append("NOT NULL")
    if f.default is not None:
        parts.append(f"DEFAULT {_sql_literal(f.default)}")
    if ref is not None:
        parts.append(f'REFERENCES "{ref.table}"("{ref.column}") ON DELETE RESTRICT')
    sql = " ".join(parts)
    if f.unique:
        sql += " UNIQUE"
    return sql


def _sql_literal(value) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def create_table_sql(
    schema: TableSchema, *, parent_table: str | None = None,
    ref_columns: dict[tuple[str, str], RefColumn] | None = None,
) -> list[str]:
    """Returns the statements to create `schema` from nothing: the table,
    its indexes, and — for non-system tables — the soft-delete + updated_at
    triggers. A `"system": true` table self-declares its own structure
    entirely (psqldb.model) and stays outside the automatic soft-delete/
    audit machinery too, same as psqldb's own _trash/_field_registry/
    _patch_history (raw SQL, never through this function, but the same
    "fully self-managed" idea)."""
    ref_columns = ref_columns or {}
    columns: list[str] = []
    for f in schema.system_fields:
        columns.append(_system_column_sql(f, parent_table=parent_table))
    for f in schema.fields:
        if f.is_column():
            columns.append(_user_column_sql(f, owner_table=schema.table, ref_columns=ref_columns))

    stmts = [
        f'CREATE TABLE IF NOT EXISTS "{schema.table}" (\n    '
        + ",\n    ".join(columns)
        + "\n)"
    ]
    stmts += index_sql(schema)
    if not schema.system:
        stmts += trigger_attach_sql(schema.table)
    return stmts


def index_sql(schema: TableSchema) -> list[str]:
    stmts = []
    for idx in schema.indexes:
        cols = ", ".join(f'"{c}"' for c in idx["fields"])
        stmts.append(f'CREATE INDEX IF NOT EXISTS "{idx["key"]}" ON "{schema.table}" ({cols})')
    return stmts


def trigger_attach_sql(table: str) -> list[str]:
    """Every non-system table gets both shared trigger functions attached.
    The functions themselves are created once, in bootstrap_sql()."""
    return [
        f'DROP TRIGGER IF EXISTS arc_set_updated_at ON "{table}"',
        f'CREATE TRIGGER arc_set_updated_at BEFORE UPDATE ON "{table}" '
        f'FOR EACH ROW EXECUTE FUNCTION arc_set_updated_at()',
        f'DROP TRIGGER IF EXISTS arc_soft_delete_to_trash ON "{table}"',
        f'CREATE TRIGGER arc_soft_delete_to_trash AFTER INSERT OR UPDATE ON "{table}" '
        f'FOR EACH ROW EXECUTE FUNCTION arc_soft_delete_to_trash()',
    ]


# ------------------------------------------------------------------------ #
# Bootstrap. Split in two so a psqldb upgrade that fixes shared trigger
# logic actually reaches an already-bootstrapped project:
#   * BOOTSTRAP_STRUCTURAL_SQL — extension + system tables. Only ever
#     needed once; migrate.build_plan only includes it when _field_registry
#     doesn't exist yet.
#   * BOOTSTRAP_FUNCTIONS_SQL — the shared trigger functions, all
#     CREATE OR REPLACE. Re-applied on EVERY `arc psqldb migrate`, so a
#     newer psqldb version's fix to arc_soft_delete_to_trash (say) lands on
#     the next migrate, not just on a brand new project.
#
# Known gap, not new: BOOTSTRAP_STRUCTURAL_SQL is CREATE TABLE IF NOT EXISTS
# only — a project that already bootstrapped BEFORE a system table's own
# shape changed here (e.g. _field_registry.ref_field, added for
# target_field/§ psqldb.migrate.resolve_ref_columns) does NOT get that
# column automatically; psqldb has no self-migration story for its OWN
# system tables yet, only for business schemas. Until that exists, an
# already-bootstrapped project picking up a psqldb change like this one
# needs the ALTER run by hand once (e.g. `ALTER TABLE _field_registry ADD
# COLUMN ref_field TEXT`).
# ------------------------------------------------------------------------ #
BOOTSTRAP_STRUCTURAL_SQL: list[str] = [
    'CREATE EXTENSION IF NOT EXISTS pgcrypto',

    # -- _patch_history: durable per-table audit log of every applied schema
    # change, from either source — a generated schema-diff migration file
    # (kind='schema') or an explicit patches/<table>.json (kind='patch').
    # Distinct from _field_registry, which only ever holds CURRENT state
    # (overwritten on every apply) — this is the history that survives it.
    """
    CREATE TABLE IF NOT EXISTS _patch_history (
        plugin      TEXT NOT NULL,
        "table"     TEXT NOT NULL,
        reference   TEXT NOT NULL,
        kind        TEXT NOT NULL CHECK (kind IN ('schema', 'patch')),
        applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (plugin, "table", kind, reference)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS _field_registry (
        id          TEXT NOT NULL,
        name        TEXT NOT NULL,
        "table"     TEXT NOT NULL,
        type        TEXT NOT NULL,
        length      INTEGER,
        reqd        BOOLEAN NOT NULL DEFAULT FALSE,
        "index"     BOOLEAN NOT NULL DEFAULT FALSE,
        "unique"    BOOLEAN NOT NULL DEFAULT FALSE,
        "default"   TEXT,
        ref_table   TEXT,
        ref_field   TEXT,
        source      TEXT NOT NULL DEFAULT 'schema' CHECK (source IN ('schema', 'patch')),
        plugin      TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY ("table", id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS _trash (
        id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "table"      TEXT NOT NULL,
        drop_type    TEXT NOT NULL CHECK (drop_type IN ('Table', 'Column', 'Row')),
        snapshot     JSONB NOT NULL,
        deleted_by   UUID,
        deleted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        restored_at  TIMESTAMPTZ
    )
    """,
]

BOOTSTRAP_FUNCTIONS_SQL: list[str] = [
    # UUID v7 (time-ordered: 48-bit millisecond timestamp prefix + random
    # tail), not Postgres's built-in gen_random_uuid() (v4, fully random) —
    # a v4 PK causes real B-tree fragmentation on every insert (each one
    # lands on a random leaf page); v7 mostly appends, like a sequential
    # int PK would, while keeping UUID's distributed-uniqueness property.
    # Hand-rolled because this targets Postgres 14 — native uuidv7() only
    # arrived in Postgres 18, and no third-party uuidv7 extension is
    # installed here either (checked pg_extension directly). Isolated in
    # its own function, re-applied every migrate like the others below, so
    # a future move to a native/extension implementation is a one-line
    # body swap with zero table DDL touched.
    """
    CREATE OR REPLACE FUNCTION arc_uuid_generate_v7() RETURNS uuid AS $$
    DECLARE
        ts_ms bytea;
        result bytea;
    BEGIN
        ts_ms := substring(int8send(floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint) FROM 3 FOR 6);
        result := ts_ms || gen_random_bytes(10);
        result := set_byte(result, 6, (get_byte(result, 6) & 15) | 112);  -- version nibble = 7
        result := set_byte(result, 8, (get_byte(result, 8) & 63) | 128);  -- variant bits
        RETURN encode(result, 'hex')::uuid;
    END;
    $$ LANGUAGE plpgsql VOLATILE
    """,

    """
    CREATE OR REPLACE FUNCTION arc_set_updated_at() RETURNS trigger AS $$
    BEGIN
        NEW.updated_at := now();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
    """,

    # AFTER trigger: the UPDATE that set _state=99 has already committed by
    # the time this runs, so deleting the row here doesn't fight the
    # original statement. Snapshots OLD (the row as it was right before the
    # delete) so a _trash row always holds real pre-delete business data —
    # recovery is then a plain re-insert, no special-casing for whatever
    # _state used to mean. Cascades to child tables via _field_registry
    # (type = 'TABLE' rows whose ref_table is this table) by setting THEIR
    # _state to 99 rather than deleting them directly — since child tables
    # carry this same trigger, the cascade recurses on its own.
    """
    CREATE OR REPLACE FUNCTION arc_soft_delete_to_trash() RETURNS trigger AS $$
    DECLARE
        child RECORD;
    BEGIN
        IF NEW._state <> 99 OR (TG_OP = 'UPDATE' AND OLD._state = 99) THEN
            RETURN NULL;
        END IF;

        -- Read updated_by via to_jsonb() rather than NEW.updated_by directly:
        -- this same function is attached to every non-system table, and a
        -- direct field reference would raise "record NEW has no field
        -- updated_by" on any future table shape that happens to lack the
        -- column (normal and child tables both have it today, but system
        -- tables never get this trigger attached at all either way).
        -- ->>'...' just yields NULL when the key is absent, so this stays
        -- correct regardless of exactly which shape a given table has.
        INSERT INTO _trash ("table", drop_type, snapshot, deleted_by, deleted_at)
        VALUES (TG_TABLE_NAME, 'Row', to_jsonb(OLD), (to_jsonb(NEW)->>'updated_by')::uuid, now());

        -- This table's own TABLE-type fields point at its children (the
        -- field lives HERE, ref_table names the child table it owns) — NOT
        -- the other way around.
        FOR child IN
            SELECT DISTINCT ref_table AS name
            FROM _field_registry
            WHERE "table" = TG_TABLE_NAME AND type = 'TABLE' AND ref_table IS NOT NULL
        LOOP
            EXECUTE format(
                'UPDATE %I SET _state = 99 WHERE parent = $1 AND _state IS DISTINCT FROM 99',
                child.name
            ) USING OLD.id;
        END LOOP;

        EXECUTE format('DELETE FROM %I WHERE id = $1', TG_TABLE_NAME) USING OLD.id;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
]


def audit_table_sql(plugin: str) -> list[str]:
    table = f"_audit_{plugin}"
    return [
        f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "table"     TEXT NOT NULL,
            row_id      UUID NOT NULL,
            changes     JSONB NOT NULL,
            changed_by  UUID,
            changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE OR REPLACE FUNCTION arc_audit_{plugin}() RETURNS trigger AS $$
        BEGIN
            INSERT INTO "{table}" ("table", row_id, changes, changed_by)
            VALUES (
                TG_TABLE_NAME,
                COALESCE(NEW.id, OLD.id),
                jsonb_build_object(
                    'before', CASE WHEN TG_OP = 'INSERT' THEN NULL ELSE to_jsonb(OLD) END,
                    'after',  CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE to_jsonb(NEW) END
                ),
                COALESCE(NEW.updated_by, OLD.updated_by)
            );
            RETURN COALESCE(NEW, OLD);
        END;
        $$ LANGUAGE plpgsql
        """,
    ]


def audit_attach_sql(table: str, plugin: str) -> list[str]:
    trigger = f"arc_audit_{plugin}"
    return [
        f'DROP TRIGGER IF EXISTS {trigger} ON "{table}"',
        f'CREATE TRIGGER {trigger} AFTER INSERT OR UPDATE OR DELETE ON "{table}" '
        f'FOR EACH ROW EXECUTE FUNCTION {trigger}()',
    ]
