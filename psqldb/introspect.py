"""
psqldb.introspect
-------------------
Reads live state back out of Postgres. Two different things live here, on
purpose:

  * `_field_registry` rows — this is what the differ (psqldb.migrate) diffs
    the declared schema AGAINST. It's the only place that remembers a
    field's stable `id`, which is what makes rename-detection possible;
    `information_schema` has no notion of it.
  * `information_schema` reads — a secondary cross-check only, to catch
    drift (someone altered a table outside of `arc psqldb migrate`). Never
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
    "the schema as of the last successful migration"."""
    rows = await conn.fetch('select * from _field_registry where "table" = $1', table)
    return [dict(r) for r in rows]


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
    for table in ("_field_registry", "_trash", "_patch_history"):
        if not await table_exists(conn, table):
            return False
    return True
