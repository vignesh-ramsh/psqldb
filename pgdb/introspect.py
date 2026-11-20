"""
pgdb.introspect
-------------------
Reads live state back out of Postgres. Two different things live here, on
purpose:

  * `_field_registry` rows — this is what the differ (pgdb.migrate) diffs
    the declared schema AGAINST. It's the only place that remembers a
    field's stable `id`, which is what makes rename-detection possible;
    `information_schema` has no notion of it.
  * `information_schema` reads — a secondary cross-check only, to catch
    drift (someone altered a table outside of `arc pgdb migrate`). Never
    the primary source of truth for diffing.
"""

from __future__ import annotations

from typing import Any


async def table_exists(conn: Any, table: str) -> bool:
    row = await conn.fetchval(
        "select exists(select 1 from information_schema.tables "
        "where table_schema = 'public' and table_name = $1)",
        table,
    )
    return bool(row)


async def registry_rows(conn: Any, table: str) -> list[dict]:
    """Every _field_registry row for `table`, keyed by field id — this is
    "the schema as of the last successful migration". On a genuinely
    fresh database, _field_registry itself doesn't exist yet — build_plan
    only QUEUES the bootstrap SQL as a plan op, it never applies it (pure
    planning, see migrate.build_plan's own docstring) — which a plain
    query here would hit as UndefinedTableError the moment a patch's
    target table is in build_plan's `will_exist` set (about to be
    created by its own plugin's schema in this same pass) rather than
    already physically existing. No registry table means no prior
    registration, by definition, so that's exactly the right answer
    here too — same reasoning table_exists() already gets right via
    information_schema, which always exists."""
    if not await table_exists(conn, "_field_registry"):
        return []
    rows = await conn.fetch('select * from _field_registry where "table" = $1', table)
    return [dict(r) for r in rows]


async def unique_together_rows(conn: Any, table: str) -> list[dict]:
    """Every _unique_together row for `table` — "the composite-unique
    groups as of the last successful migration", the registry
    pgdb.migrate diffs a schema's declared groups against, the same role
    registry_rows() plays for individual fields. Same not-bootstrapped-yet
    guard as registry_rows() and for the identical reason — see that
    function's own docstring."""
    if not await table_exists(conn, "_unique_together"):
        return []
    rows = await conn.fetch('select * from _unique_together where "table" = $1', table)
    return [dict(r) for r in rows]


async def constraint_exists(conn: Any, table: str, constraint_name: str) -> bool:
    """Whether a constraint (of any kind) already exists on `table` under
    this exact name — used to make adding a unique_together group
    idempotent. Unlike `CREATE INDEX IF NOT EXISTS`, Postgres has no `ADD
    CONSTRAINT IF NOT EXISTS`; this is the check pgdb.migrate runs first
    so a re-migrate with an unchanged group renders no ALTER TABLE at all,
    rather than one that would fail with a duplicate-constraint error."""
    row = await conn.fetchval(
        "select exists(select 1 from pg_constraint c "
        "join pg_class t on t.oid = c.conrelid "
        "where t.relname = $1 and c.conname = $2)",
        table,
        constraint_name,
    )
    return bool(row)


async def live_columns(conn: Any, table: str) -> dict[str, dict]:
    """information_schema view of `table`'s actual columns, keyed by column
    name — used only for drift warnings (§ module docstring), never for
    diffing itself."""
    rows = await conn.fetch(
        "select column_name, data_type, is_nullable, character_maximum_length "
        "from information_schema.columns "
        "where table_schema = 'public' and table_name = $1",
        table,
    )
    return {r["column_name"]: dict(r) for r in rows}


async def bootstrap_applied(conn: Any) -> bool:
    """True only if every structural bootstrap table exists — not just
    _field_registry. A project that pre-dates a bootstrap-table rename (or
    addition) would otherwise never get the missing one created, since the
    old heuristic treated "_field_registry exists" as "bootstrap fully
    ran". The structural SQL itself is all CREATE ... IF NOT EXISTS, so
    re-running it whenever any one of these is missing is always safe."""
    for table in (
        "_field_registry",
        "_trash",
        "_patch_history",
        "_unique_together",
        "_index_registry",
        "_routines_history",
    ):
        if not await table_exists(conn, table):
            return False
    return True


async def index_registry_rows(conn: Any, table: str) -> list[dict]:
    """Every _index_registry row for `table` — the named indexes pgdb created
    as of the last successful migration. Empty before the registry exists."""
    if not await table_exists(conn, "_index_registry"):
        return []
    rows = await conn.fetch('select * from _index_registry where "table" = $1', table)
    return [dict(r) for r in rows]


async def live_index(conn: Any, key: str) -> dict | None:
    """The real index named `key` (index names are unique per schema, not per
    table): {"table", "columns" (in order), "method"}, or None when there is
    none. Expression columns come back as None entries."""
    row = await conn.fetchrow(
        "select t.relname as tbl, am.amname as method, "
        "array(select a.attname::text from unnest(i.indkey::int2[]) with ordinality k(attnum, ord) "
        "      left join pg_attribute a on a.attrelid = t.oid and a.attnum = k.attnum "
        "      order by k.ord) as cols "
        "from pg_class ic join pg_index i on i.indexrelid = ic.oid "
        "join pg_class t on t.oid = i.indrelid "
        "join pg_namespace n on n.oid = ic.relnamespace "
        "join pg_am am on am.oid = ic.relam "
        "where n.nspname = 'public' and ic.relname = $1",
        key,
    )
    if row is None:
        return None
    return {"table": row["tbl"], "columns": list(row["cols"]), "method": row["method"]}


async def trigger_exists(conn: Any, table: str, trigger: str) -> bool:
    """Whether `table` carries a (non-internal) trigger called `trigger`."""
    row = await conn.fetchval(
        "select exists(select 1 from pg_trigger g join pg_class t on t.oid = g.tgrelid "
        "join pg_namespace n on n.oid = t.relnamespace "
        "where n.nspname = 'public' and t.relname = $1 and g.tgname = $2 and not g.tgisinternal)",
        table,
        trigger,
    )
    return bool(row)


async def single_column_constraints(
    conn: Any, table: str, columns: list[str], kind: str
) -> list[str]:
    """Names of the single-column constraints of `kind` ('u' unique, 'f'
    foreign key) on any of `columns` of `table`. Found by COLUMN, not by the
    name pgdb would generate: a constraint keeps the name it was created
    under when its column is later renamed."""
    rows = await conn.fetch(
        "select c.conname from pg_constraint c "
        "join pg_class t on t.oid = c.conrelid "
        "join pg_namespace n on n.oid = t.relnamespace "
        "where n.nspname = 'public' and t.relname = $1 and c.contype::text = $3 "
        "and cardinality(c.conkey) = 1 "
        "and (select a.attname::text from pg_attribute a "
        "     where a.attrelid = t.oid and a.attnum = c.conkey[1]) = any($2::text[]) "
        "order by c.conname",
        table,
        columns,
        kind,
    )
    return [r["conname"] for r in rows]


async def patch_registry_rows(conn: Any) -> list[dict]:
    """Every _field_registry row that a patch (not a schema) declared, across
    all tables. Empty before the registry exists."""
    if not await table_exists(conn, "_field_registry"):
        return []
    rows = await conn.fetch("select * from _field_registry where source = 'patch'")
    return [dict(r) for r in rows]
