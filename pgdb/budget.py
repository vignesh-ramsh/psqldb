"""
pgdb.budget
-----------------
PostgreSQL's connection limits, and how a process's pool size is chosen from them.

Every ARC process owns its own pool, so a deployment can open (processes x pool size)
connections, and PostgreSQL refuses the ones past its limit. The limit is not one number: an
ordinary role may use at most the smallest of

    max_connections - superuser_reserved_connections
    the role's own CONNECTION LIMIT      (ALTER ROLE ... CONNECTION LIMIT n)
    the database's CONNECTION LIMIT      (ALTER DATABASE ... CONNECTION LIMIT n)

(verified against a real server: with a role limit of 12, 15 connection attempts gave 12
accepted and 3 refused with `too many connections for role`). Behind PgBouncer, or on a managed
database that hides these, none of them is the real limit — `pgdb_connection_budget` states it.

The splitting itself is plain arithmetic in `arc.sizing`; this module only reads the server and
decides where a process's number comes from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import asyncpg

from arc import sizing

#: Set by `arc deploy` (and `arc run`) for every process they start: which kind of process this is.
ROLE_ENV = "ARC_PROCESS_ROLE"
#: JSON written by `arc deploy`: the plan it made (pool size per role, process counts, the limit).
PLAN_KEY = "pgdb_connection_plan"
#: The same plan handed straight to child processes by `arc run`, which plans for the
#: processes IT starts and writes nothing down. Wins over the stored plan.
PLAN_ENV = "ARC_CONNECTION_PLAN"
#: Total connections ARC may use, when the server's own numbers aren't the real limit.
BUDGET_KEY = "pgdb_connection_budget"
#: Connections kept free for migrations, `arc console`, backups, an administrator's psql.
RESERVE_KEY = "pgdb_connection_reserve"
#: Per-role override of the pool size, e.g. pgdb_pool_max_size_lineup_worker (never changed by a deploy).
ROLE_POOL_KEY_PREFIX = "pgdb_pool_max_size_"
#: What resolve_pool_size reports when the number came from the plan `arc deploy` made.
PLAN_SOURCE = "the deployment's connection plan"

LIMITS_SQL = """
select current_setting('max_connections')::int                                   as max_connections,
       current_setting('superuser_reserved_connections')::int                    as reserved,
       (select rolconnlimit from pg_roles where rolname = current_user)          as role_limit,
       (select datconnlimit from pg_database where datname = current_database()) as db_limit
"""


@dataclass(frozen=True)
class ServerLimits:
    max_connections: int
    reserved: int  # kept for superusers
    role_limit: int  # -1: none
    db_limit: int  # -1: none

    @property
    def usable(self) -> int:
        """What an ordinary role in this database may open: the smallest applicable limit."""
        limits = [self.max_connections - self.reserved]
        limits += [n for n in (self.role_limit, self.db_limit) if n >= 0]
        return min(limits)


IN_USE_SQL = """
select count(*) from pg_stat_activity
where datname = current_database() and backend_type = 'client backend'
"""


async def fetch_in_use(conn: asyncpg.Connection) -> int:
    """Connections open to this database right now (everyone's, this one included)."""
    return int(await conn.fetchval(IN_USE_SQL))


async def inspect_dsn(dsn: str, *, timeout: float = 5.0) -> tuple[ServerLimits, int]:
    """The server's limits and how many connections are open, over one short-lived connection
    (`arc status`, `arc doctor --deep`)."""
    conn = await asyncpg.connect(dsn, timeout=timeout)
    try:
        return await fetch_limits(conn), await fetch_in_use(conn)
    finally:
        await conn.close()


async def fetch_limits(conn: asyncpg.Connection) -> ServerLimits:
    row = await conn.fetchrow(LIMITS_SQL)
    return ServerLimits(row["max_connections"], row["reserved"], row["role_limit"], row["db_limit"])


async def fetch_limits_for_dsn(dsn: str, *, timeout: float = 5.0) -> ServerLimits:
    """One short-lived connection, used by `arc deploy` and by a process before it opens its pool."""
    conn = await asyncpg.connect(dsn, timeout=timeout)
    try:
        return await fetch_limits(conn)
    finally:
        await conn.close()


def role_pool_key(role: str) -> str:
    return ROLE_POOL_KEY_PREFIX + role.replace("-", "_")


def usable_connections(limits: ServerLimits, budget_override: int | None) -> int:
    """The explicit budget replaces what the server reports (it is for the cases where the
    server's numbers aren't the real limit: a pooler in front, a managed database)."""
    return budget_override if budget_override and budget_override > 0 else limits.usable


def _int(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def resolve_pool_size(
    role: str | None,
    *,
    get: Callable[[str], Any],
    plan: sizing.ConnectionPlan | None,
    global_size: int,
) -> tuple[int, str]:
    """(pool max size, where it came from) for this process.

    1. the operator's own `pgdb_pool_max_size_<role>` — never overridden by a deployment;
    2. the plan `arc deploy` made for this role;
    3. `pgdb_pool_max_size` (default 10) — for a process with no role (a CLI command) or no plan
       (`arc run`, a project never deployed)."""
    if role:
        key = role_pool_key(role)
        explicit = _int(get(key))
        if explicit is not None:
            return explicit, f"setting {key}"
        if plan is not None and role in plan.pools:
            return plan.pools[role], PLAN_SOURCE
    return global_size, "setting pgdb_pool_max_size"


def role_from_environment() -> str | None:
    return os.environ.get(ROLE_ENV) or None


def plan_text(stored_setting: Any) -> str:
    """The plan this process should use: the one `arc run` gave it, else the one `arc deploy` stored."""
    return os.environ.get(PLAN_ENV) or stored_setting or ""
