"""
pgdb.dump
-----------------
`pg_dump` / `pg_restore` around a DSN: what `arc pgdb backup` / `restore` use for a database or a
slice of it, and what `arc backup` (the kernel's full backup) uses for the whole database.

Real client tools, not a reimplementation. The one thing worth doing here is how they are called: the
DSN is split so that the password travels in the child's environment (PGPASSWORD), never in argv,
where any local user can read it out of /proc/<pid>/cmdline or `ps`.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlparse

import asyncpg


class DumpError(RuntimeError):
    """A pg_dump / pg_restore run failed, or the tools are missing."""


def connection_args(dsn: str) -> tuple[list[str], str, dict[str, str]]:
    """Splits `dsn` into (connection args, dbname, environment) for pg_dump/pg_restore, instead of
    handing either tool the raw DSN as a single argv element.

    argv is world-readable for the life of the process - any local user can read the full DSN,
    password included, out of /proc/<pid>/cmdline or `ps aux` - but a process's own environment is
    not. host/port/user/dbname aren't secret, so only the password moves to PGPASSWORD (the standard
    env var every libpq client tool already reads for exactly this). `unquote()` undoes this
    project's own DSN construction, which quote_plus's user/password so `@`/`:`/`/` in either can't
    be misparsed as DSN structure - urlparse's .username/.password hand back the still-encoded form."""
    parsed = urlparse(dsn)
    args: list[str] = []
    if parsed.hostname:
        args += ["-h", parsed.hostname]
    if parsed.port:
        args += ["-p", str(parsed.port)]
    if parsed.username:
        args += ["-U", unquote(parsed.username)]
    env = dict(os.environ)
    if parsed.password:
        env["PGPASSWORD"] = unquote(parsed.password)
    return args, parsed.path.lstrip("/"), env


def require_tools(*names: str) -> None:
    """Raise DumpError naming whichever of `names` (pg_dump, pg_restore) is not on PATH."""
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise DumpError(
            f"{' and '.join('`' + n + '`' for n in missing)} not found on PATH - install the "
            f"PostgreSQL client tools (e.g. `apt install postgresql-client`)."
        )


def _run(argv: list[str], env: dict[str, str], what: str) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line.strip() for line in (result.stderr or result.stdout).splitlines() if line.strip()]
        # the lines that say what went wrong, not the "errors ignored on restore: 3" summary after them
        errors = [line for line in lines if "error" in line.lower() and "ignored on restore" not in line]
        detail = errors[:4] or lines[-3:]
        raise DumpError(f"{what} failed (exit {result.returncode}): " + (" | ".join(detail) or "no output"))
    return result


def dump(dsn: str, out: Path, *, tables: Sequence[str] = ()) -> None:
    """pg_dump the database (or only `tables`) to `out` in the custom format (compressed, and what
    pg_restore needs for selective restores)."""
    require_tools("pg_dump")
    args, dbname, env = connection_args(dsn)
    argv = ["pg_dump", "-Fc", "-f", str(out), *args, dbname]
    for table in tables:
        argv += ["-t", table]
    _run(argv, env, "pg_dump")


def restore(
    dsn: str,
    dump_file: Path,
    *,
    tables: Sequence[str] = (),
    replace: bool = True,
    portable: bool = False,
) -> None:
    """pg_restore `dump_file` into the database `dsn` names.

    replace: drop each object first (--clean --if-exists) - what `arc pgdb restore` has always done
             to put a table back. False: restore into an empty database, and fail on a clash.
    portable: ignore the original owners and grants (--no-owner --no-privileges) and stop at the
             first error (--exit-on-error) - for restoring a whole backup somewhere else, where the
             role that owned things may not exist and a half-restored database must not look done."""
    require_tools("pg_restore")
    args, dbname, env = connection_args(dsn)
    argv = ["pg_restore", *args, "-d", dbname]
    if replace:
        argv += ["--clean", "--if-exists"]
    if portable:
        argv += ["--no-owner", "--no-privileges", "--exit-on-error"]
    for table in tables:
        argv += ["-t", table]
    argv.append(str(dump_file))
    _run(argv, env, "pg_restore")


def describe(dump_file: Path) -> str:
    """The dump's table of contents (`pg_restore --list`). Raises DumpError if the file is not a
    readable pg_dump custom-format archive - the cheapest check that a backup is not garbage."""
    require_tools("pg_restore")
    return _run(["pg_restore", "--list", str(dump_file)], dict(os.environ), "pg_restore --list").stdout


_RELATIONS_SQL = """
SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_toast%'
  AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
"""


def is_empty(dsn: str) -> bool:
    """Does the database hold no tables, views or sequences of its own? A restore target should."""

    async def count() -> int:
        conn = await asyncpg.connect(dsn, timeout=10)
        try:
            return int(await conn.fetchval(_RELATIONS_SQL))
        finally:
            await conn.close()

    return asyncio.run(count()) == 0
