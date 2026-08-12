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

    table: str  # physical target table
    column: str  # physical target column ("id", or a declared unique field's name)
    sql_type: str  # that column's own SQL type — the REFERENCING column must match it


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
        # Stores the acting user's EMAIL, not a UUID — readable without a
        # join to whatever users table (if any) exists. Nullable, no FK —
        # psqldb doesn't know a "users" table exists (§3.3), which is also
        # exactly why a UUID here was never resolvable at the psqldb layer.
        return f'"{f.name}" TEXT'
    if f.id == "_state":
        return '"_state" INTEGER NOT NULL DEFAULT 0'
    raise AssertionError(f"unhandled system field id {f.id!r}")


def _user_column_sql(
    f: Field, *, owner_table: str, ref_columns: dict[tuple[str, str], RefColumn]
) -> str:
    if f.primary_key:
        # Same shape a normal table's auto-injected id gets (_system_column_sql's
        # "_id" branch above) — just self-declared, since a "system": true table
        # has no auto-injected system_fields at all (psqldb.model). A PRIMARY KEY
        # is already NOT NULL UNIQUE, and arc_uuid_generate_v7() is a function
        # call, not a literal, so this deliberately skips the generic
        # required/default/unique rendering below (_sql_literal would wrongly
        # quote a function call as a string).
        return f'"{f.name}" UUID PRIMARY KEY DEFAULT arc_uuid_generate_v7()'
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
    schema: TableSchema,
    *,
    parent_table: str | None = None,
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
    # Table-level constraints mixed into the same paren'd list Postgres
    # already accepts column defs in — a brand-new table gets its
    # unique_together groups inline here; an EXISTING table picking one up
    # on a later migrate goes through unique_together_sql's ALTER TABLE
    # instead (see its own docstring for why: no "ADD CONSTRAINT IF NOT
    # EXISTS" exists in Postgres, unlike CREATE INDEX).
    for ut in schema.unique_together:
        cols = ", ".join(f'"{c}"' for c in ut["fields"])
        columns.append(f'CONSTRAINT "{ut["key"]}" UNIQUE ({cols})')

    stmts = [
        f'CREATE TABLE IF NOT EXISTS "{schema.table}" (\n    ' + ",\n    ".join(columns) + "\n)"
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


def unique_together_sql(table: str, group: dict) -> str:
    """ADD CONSTRAINT for one unique_together group on an EXISTING table —
    used when a later migrate adds a group that wasn't there before (a
    brand-new table gets its groups rendered inline in create_table_sql
    instead, since CREATE TABLE can just list them as table constraints).

    Unlike index_sql's `CREATE INDEX IF NOT EXISTS`, Postgres has no `ADD
    CONSTRAINT IF NOT EXISTS` — this always renders the statement
    unconditionally; psqldb.migrate only calls it for a group
    psqldb.introspect.constraint_exists() has already confirmed is
    genuinely new, so idempotency is the caller's job, not this
    function's (same "renders, never decides" split every other function
    in this module already follows)."""
    cols = ", ".join(f'"{c}"' for c in group["fields"])
    return f'ALTER TABLE "{table}" ADD CONSTRAINT "{group["key"]}" UNIQUE ({cols})'


def drop_unique_together_sql(table: str, key: str) -> str:
    return f'ALTER TABLE "{table}" DROP CONSTRAINT IF EXISTS "{key}"'


def trigger_attach_sql(table: str) -> list[str]:
    """Every non-system table gets both shared trigger functions attached.
    The functions themselves are created once, in bootstrap_sql()."""
    return [
        f'DROP TRIGGER IF EXISTS arc_set_updated_at ON "{table}"',
        f'CREATE TRIGGER arc_set_updated_at BEFORE UPDATE ON "{table}" '
        f"FOR EACH ROW EXECUTE FUNCTION arc_set_updated_at()",
        f'DROP TRIGGER IF EXISTS arc_soft_delete_to_trash ON "{table}"',
        f'CREATE TRIGGER arc_soft_delete_to_trash AFTER INSERT OR UPDATE ON "{table}" '
        f"FOR EACH ROW EXECUTE FUNCTION arc_soft_delete_to_trash()",
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
# COLUMN ref_field TEXT`). `precision`/`scale` (added alongside the DECIMAL
# diffing fix, psqldb.migrate._diff_table) are the same class of change —
# an already-bootstrapped project needs `ALTER TABLE _field_registry ADD
# COLUMN precision INTEGER, ADD COLUMN scale INTEGER` run by hand once.
# Existing DECIMAL fields already in the registry will read back NULL for
# both until their next migrate re-registers them — the diff below treats
# that as "unknown, therefore changed" (same safe-by-default posture as
# every other destructive classification here), so the very next migrate
# after this upgrade will show a one-time, harmless ALTER COLUMN TYPE
# review op for each of them even though nothing really changed.
#
# The reverse also happens — a column can be REMOVED here without an
# automatic DROP either. `_field_registry."index"` (a BOOLEAN, meant to
# flag an indexed field) was dead from the start: registry_upsert_sql never
# wrote it (its own INSERT column list never named it, so it stayed FALSE
# forever) and nothing ever read it — named indexes are tracked only as the
# literal `CREATE INDEX` statements themselves, deliberately never diffed
# against a registry column (this module's own docstring: index removal is
# a documented no-op, not a gap). Removed here as dead-code cleanup, not a
# behavior change; an already-bootstrapped project keeps the orphaned
# column until an operator runs `ALTER TABLE _field_registry DROP COLUMN
# "index"` by hand — same one-time-manual-ALTER story as every other
# _field_registry shape change above, just in the opposite direction.
# ------------------------------------------------------------------------ #
BOOTSTRAP_STRUCTURAL_SQL: list[str] = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
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
        precision   INTEGER,
        scale       INTEGER,
        reqd        BOOLEAN NOT NULL DEFAULT FALSE,
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
        id           UUID PRIMARY KEY DEFAULT arc_uuid_generate_v7(),
        "table"      TEXT NOT NULL,
        drop_type    TEXT NOT NULL CHECK (drop_type IN ('Table', 'Column', 'Row')),
        snapshot     JSONB NOT NULL,
        deleted_by   TEXT,
        deleted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        restored_at  TIMESTAMPTZ
    )
    """,
    # -- _unique_together: "the composite-unique groups as of the last
    # successful migration" — the registry psqldb.migrate diffs a schema's
    # declared unique_together list against, the same role _field_registry
    # plays for individual fields. A separate table rather than more
    # _field_registry columns because a group spans MULTIPLE fields, not
    # one row per field; `fields` is stored as a real Postgres TEXT[] (not
    # JSONB) since nothing here ever needs to query inside it, only compare
    # it whole. Added here rather than via an ALTER on an already-
    # bootstrapped project (contrast _field_registry's own precision/scale
    # note above): bootstrap_applied() checks for this table BY NAME, so an
    # existing project simply re-runs this CREATE TABLE IF NOT EXISTS block
    # on its very next migrate — no manual ALTER needed, since this is a
    # whole new table, not a new column on one that already exists.
    """
    CREATE TABLE IF NOT EXISTS _unique_together (
        "table"     TEXT NOT NULL,
        "key"       TEXT NOT NULL,
        fields      TEXT[] NOT NULL,
        plugin      TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY ("table", "key")
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
    # Only bumps updated_at when the row actually changed — psqldb.update()
    # has no dirty-check of its own (it only guards "zero columns to set at
    # all", psqldb/__init__.py), so a caller that saves a row with the
    # SAME values it already has used to still issue a real UPDATE that
    # unconditionally bumped updated_at (and, on an audited table, wrote a
    # before==after row to _audit_* — see arc_audit_{plugin} below, which
    # this fix also protects). Every OTHER caller of psqldb.update() besides
    # the one place that added its own client-side dirty-check would
    # reproduce that exact bug; fixing it here, in the one function every
    # non-system table's UPDATE already goes through, protects all of them
    # at once, at zero extra cost on the genuine-change path (BEFORE
    # triggers already see both OLD and NEW for free — no extra query).
    # Compares everything EXCEPT updated_at itself (which would otherwise
    # always look "different" — OLD.updated_at is whatever it was last set
    # to, NEW.updated_at hasn't been assigned yet at this point) using
    # jsonb's `-` key-removal operator.
    """
    CREATE OR REPLACE FUNCTION arc_set_updated_at() RETURNS trigger AS $$
    BEGIN
        IF to_jsonb(OLD) - 'updated_at' IS DISTINCT FROM to_jsonb(NEW) - 'updated_at' THEN
            NEW.updated_at := now();
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
    """,
    # AFTER trigger: it runs inside the SAME transaction as the UPDATE that
    # set _state=99 (an AFTER ROW trigger fires after that statement's row
    # change, NOT after commit — which is exactly what lets a rolled-back
    # batch take its trash snapshots down with it), but after the row change
    # is complete, so deleting the row here doesn't fight the original
    # statement. Snapshots OLD (the row as it was right before the
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
        VALUES (TG_TABLE_NAME, 'Row', to_jsonb(OLD), to_jsonb(NEW)->>'updated_by', now());

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
            id          UUID PRIMARY KEY DEFAULT arc_uuid_generate_v7(),
            "table"     TEXT NOT NULL,
            row_id      UUID NOT NULL,
            changes     JSONB NOT NULL,
            changed_by  TEXT,
            changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        # admin.audit_api.list_audit_entries's real query shape: WHERE
        # row_id = $1 ORDER BY id DESC LIMIT $2 (id, not changed_at — id
        # is UUIDv7, arc_uuid_generate_v7() above, already chronological).
        # Without this, that query is a full seq scan of the whole audit
        # table on every "Row Preview" open, and only gets worse as it
        # grows — this table has no TableSchema so it never goes through
        # the normal patches/index mechanism; IF NOT EXISTS + re-run on
        # every migrate (maintenance=True in migrate.py) is how an
        # already-existing "_audit_{plugin}" table picks this up too, not
        # just a freshly created one.
        f'CREATE INDEX IF NOT EXISTS "idx_audit_{plugin}_row_id" ON "{table}" (row_id, id DESC)',
        f"""
        CREATE OR REPLACE FUNCTION arc_audit_{plugin}() RETURNS trigger AS $$
        DECLARE
            new_json jsonb := to_jsonb(NEW);
            old_json jsonb := to_jsonb(OLD);
        BEGIN
            -- INSERT writes no audit row at all — a product decision (not
            -- the original "INSERT is never a no-op" reasoning this
            -- function used to state, which was true but beside the
            -- point): a row's creation is already on the row itself
            -- (created_by/created_at), and the audit trail is meant to
            -- start once something is actually CHANGED, not restate the
            -- row's own first values back at it. A genuinely no-op UPDATE
            -- (every column identical) also writes nothing — arc_set_
            -- updated_at (a BEFORE trigger, always fires first) already
            -- leaves updated_at itself untouched in that same case, so
            -- old_json = new_json here means the whole row, updated_at
            -- included, is unchanged. DELETE has no such thing as a
            -- no-op — always recorded.
            IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND old_json = new_json) THEN
                RETURN NEW;
            END IF;

            -- changed_by goes through the jsonb representation, not a direct
            -- NEW.updated_by/OLD.updated_by struct reference: this function is
            -- shared across every audited table for the plugin, and a
            -- "system": true table (psqldb.model) self-declares its own
            -- columns — it has no auto-injected updated_by at all. ->>'key' on
            -- jsonb degrades to NULL when the key's absent instead of raising
            -- "record has no field", so this works for both table shapes
            -- without the function needing to know which one it's on.
            --
            -- 'before' is always old_json, unconditionally, now that INSERT
            -- returns above before ever reaching this statement — only
            -- UPDATE (a real change) and DELETE still get here, and both
            -- always have a real OLD row.
            INSERT INTO "{table}" ("table", row_id, changes, changed_by)
            VALUES (
                TG_TABLE_NAME,
                COALESCE(NEW.id, OLD.id),
                jsonb_build_object(
                    'before', old_json,
                    'after',  CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE new_json END
                ),
                COALESCE(
                    new_json->>'updated_by', new_json->>'created_by',
                    old_json->>'updated_by', old_json->>'created_by'
                )
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
        f"FOR EACH ROW EXECUTE FUNCTION {trigger}()",
    ]
