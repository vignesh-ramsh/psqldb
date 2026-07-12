"""
psqldb.cli — `arc psqldb ...` commands.

Mounted onto the core `arc` CLI via the `arc.plugins.cli` entry point (see
arc.plugin_cli in the kernel) — this file is never imported by the kernel
directly, only discovered by name.

Deliberately independent of arc.boot(): these are raw operational/debug
tools (the same job `bench mariadb` does for a Frappe site) — a human
poking at the database directly, not the application's own async runtime
path. So they read the DSN straight off disk via SettingsManager rather
than requiring a full boot.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
import typer
from rich.console import Console

from arc.runtime import find_project_root
from arc.settings import SettingsManager

from . import DSN_KEY, migrate

app = typer.Typer(help="Commands for the psqldb provider.")
trash_app = typer.Typer(help="Recover documents from _trash.")
app.add_typer(trash_app, name="trash")
console = Console()
err_console = Console(stderr=True, style="bold red")


def _dsn() -> str:
    root = find_project_root()
    if root is None:
        err_console.print(
            "Not inside an ARC project (no .arc/arc.toml found here or in any parent)."
        )
        raise typer.Exit(code=1)

    mgr = SettingsManager(root / ".arc")
    dsn = mgr.get(DSN_KEY, reveal=True)
    if dsn is None:
        err_console.print(
            f"'{DSN_KEY}' is not set. Run: "
            f"arc settings set {DSN_KEY} postgresql://user:pass@host:5432/dbname --secret"
        )
        raise typer.Exit(code=1)
    return dsn


@app.command()
def status() -> None:
    """Check connectivity to the configured Postgres database."""
    dsn = _dsn()
    parsed = urlparse(dsn)

    async def _check() -> tuple[str, float]:
        start = time.monotonic()
        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            version = await conn.fetchval("select version()")
        finally:
            await conn.close()
        return version, time.monotonic() - start

    try:
        version, elapsed = asyncio.run(_check())
    except Exception as exc:
        err_console.print(
            f"psqldb: FAILED to connect to "
            f"{parsed.hostname}:{parsed.port or 5432} — {exc}"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold green]psqldb: OK[/bold green] ({elapsed * 1000:.0f}ms)")
    console.print(f"  host:     {parsed.hostname}:{parsed.port or 5432}")
    console.print(f"  database: {parsed.path.lstrip('/')}")
    console.print(f"  server:   {version}")


@app.command()
def connect() -> None:
    """Drop into an interactive psql shell against the configured database — same job as `bench mariadb`."""
    dsn = _dsn()
    if shutil.which("psql") is None:
        err_console.print(
            "`psql` was not found on PATH. Install the PostgreSQL client "
            "(e.g. `apt-get install postgresql-client`) and try again."
        )
        raise typer.Exit(code=1)

    parsed = urlparse(dsn)
    dbname = parsed.path.lstrip("/") or parsed.username or "postgres"
    argv = [
        "psql",
        "-h", parsed.hostname or "localhost",
        "-p", str(parsed.port or 5432),
        "-U", parsed.username or "postgres",
        "-d", dbname,
    ]

    env = os.environ.copy()
    if parsed.password:
        # PGPASSWORD, not a -p<password> flag — a CLI arg would be visible to
        # every other user on the box via `ps`; the env var is not.
        env["PGPASSWORD"] = parsed.password

    console.print(f"[dim]$ {' '.join(argv)}[/dim]")
    os.execvpe("psql", argv, env)  # replace this process — real TTY, correct signal handling


# ------------------------------------------------------------------------ #
# plan / migrate — arc.boot() first (so every business plugin's own
# register() has called arc.psqldb.register_model(...)), then diff each
# registered schema against the live DB. Same "boot to collect declarations"
# pattern as `arc gateway routes`.
# ------------------------------------------------------------------------ #
def _boot() -> "PsqlDbProvider":  # noqa: F821 - forward ref, avoids importing arc's own psqldb attribute at module load
    import arc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", arc.ArcAdvisory)
        arc.boot()
    return arc.psqldb


@contextlib.contextmanager
def _friendly_errors():
    """Schema/patch problems (bad JSON, ownership collisions, a table that
    already exists under a different owner, ...) are user mistakes to fix,
    not internal failures — report them as a clean one-line error, not a
    Python traceback."""
    from .migrate import MigrationError
    from .model import SchemaError

    try:
        yield
    except (MigrationError, SchemaError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc


def _root_or_exit() -> Path:
    root = find_project_root()
    if root is None:
        err_console.print("Not inside an ARC project (no .arc/arc.toml found here or in any parent).")
        raise typer.Exit(code=1)
    return root


async def _build_plan(plugin: str | None, table: str | None) -> tuple[object, migrate.MigrationPlan]:
    provider = _boot()
    schemas, patches = provider.schemas(), provider.patches()
    if plugin:
        schemas = [s for s in schemas if s.plugin == plugin]
        patches = [p for p in patches if p.plugin == plugin]
        if not schemas and not patches:
            err_console.print(f"No schemas or patches registered for plugin '{plugin}'.")
            raise typer.Exit(code=1)
    await provider.open()
    try:
        async with provider.acquire() as conn:
            plan = await migrate.build_plan(conn, provider.schemas(), provider.patches(), only_table=table)
            if plugin:
                plan.ops = [op for op in plan.ops if op.plugin == plugin or op.table == "_bootstrap"]
    finally:
        await provider.close()
    return provider, plan


def _print_plan(plan: migrate.MigrationPlan) -> None:
    if plan.is_empty():
        console.print("[dim]No schema changes — the live database already matches every registered schema/patch.[/dim]")
    else:
        for table, ops in plan.by_table().items():
            console.print(f"[bold]{table}[/bold]")
            for op in ops:
                style = "bold red" if op.destructive else "green"
                tag = "DESTRUCTIVE" if op.destructive else "safe"
                console.print(f"  [{style}][{tag}][/{style}] ({op.source}) {op.description}")
    for warning in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command()
def plan(
    plugin: str = typer.Option(None, "-p", "--plugin", help="Only show changes for this plugin's schemas/patches."),
    table: str = typer.Option(None, "-t", "--table", help="Only show changes for this table."),
) -> None:
    """Preview what `arc psqldb migrate` would do — never touches the database."""
    _root_or_exit()
    with _friendly_errors():
        _provider, the_plan = asyncio.run(_build_plan(plugin, table))
        _print_plan(the_plan)


@app.command()
def migrate_(
    plugin: str = typer.Option(None, "-p", "--plugin", help="Only migrate this plugin's schemas/patches."),
    table: str = typer.Option(None, "-t", "--table", help="Only migrate this table."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt (for CI/non-interactive use)."),
) -> None:
    """Diff every registered schema AND patch against the live DB, show the
    plan, and — after confirmation — apply it. Always shows the plan first,
    even with --yes; destructive changes (drops, type/NOT NULL tightening)
    are always recoverable from _trash, but are still shown in red."""
    root = _root_or_exit()

    async def _run():
        provider = _boot()
        schemas, patches = provider.schemas(), provider.patches()
        target_plugins = {s.plugin for s in schemas if not plugin or s.plugin == plugin} | \
                          {p.plugin for p in patches if not plugin or p.plugin == plugin}
        reference = migrate.migration_reference()
        await provider.open()
        try:
            async with provider.acquire() as conn:
                the_plan = await migrate.build_plan(conn, schemas, patches, only_table=table)
                if plugin:
                    the_plan.ops = [op for op in the_plan.ops if op.plugin == plugin or op.table == "_bootstrap"]
                _print_plan(the_plan)
                if the_plan.is_empty():
                    return
                if not yes and not typer.confirm("Proceed?", default=False):
                    console.print("[dim]Aborted — nothing applied.[/dim]")
                    raise typer.Exit(code=1)
                await migrate.apply_plan(conn, the_plan, reference=reference)
            for plugin_name in sorted(target_plugins):
                plugin_ops_plan = migrate.MigrationPlan(
                    ops=[op for op in the_plan.ops if op.plugin == plugin_name]
                )
                if plugin_ops_plan.ops:
                    path = migrate.write_migration_file(
                        root / "plugins" / plugin_name, plugin_name, plugin_ops_plan, reference
                    )
                    console.print(f"[dim]wrote {path}[/dim]")
        finally:
            await provider.close()

    with _friendly_errors():
        asyncio.run(_run())
    console.print("[bold green]Migration complete.[/bold green]")


app.command(name="migrate")(migrate_)


@app.command()
def history(
    plugin: str = typer.Option(None, "-p", "--plugin", help="Only show this plugin's history."),
    table: str = typer.Option(None, "-t", "--table", help="Only show this table's history."),
) -> None:
    """Show what's recorded in _patch_history — every applied schema change
    (from a schema OR a patch), per table, with when and via which
    generated file. Unlike _field_registry (current state only, overwritten
    on every apply), this is the durable history."""
    dsn = _dsn()

    async def _run():
        conn = await asyncpg.connect(dsn)
        try:
            query = 'SELECT plugin, "table", reference, kind, applied_at FROM _patch_history'
            clauses, params = [], []
            if plugin:
                params.append(plugin)
                clauses.append(f"plugin = ${len(params)}")
            if table:
                params.append(table)
                clauses.append(f'"table" = ${len(params)}')
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY applied_at"
            return await conn.fetch(query, *params)
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    if not rows:
        console.print("[dim]No history recorded yet — run `arc psqldb migrate`.[/dim]")
        return
    for r in rows:
        console.print(f"{r['applied_at']}  {r['plugin']:<16} {r['table']:<20} {r['kind']:<6} {r['reference']}")


@app.command()
def clear(
    table: str = typer.Option(None, "-t", "--table", help="Clear this one table."),
    plugin: str = typer.Option(None, "-p", "--plugin", help="Clear every table this plugin owns."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete every row from a table (or every table a plugin owns) — via
    the same soft-delete-to-trash path as a single delete, so it's
    recoverable and cascades to child tables. Requires -t or -p; never
    clears the whole database unscoped."""
    if not table and not plugin:
        err_console.print("Specify -t/--table or -p/--plugin — `clear` never runs unscoped.")
        raise typer.Exit(code=1)

    async def _run():
        provider = _boot()
        tables = {table} if table else {s.table for s in provider.schemas() if s.plugin == plugin}
        if plugin and not tables:
            err_console.print(f"No schemas registered for plugin '{plugin}'.")
            raise typer.Exit(code=1)

        console.print(f"About to clear all rows from: {', '.join(sorted(tables))}")
        if not yes and not typer.confirm("Proceed?", default=False):
            console.print("[dim]Aborted — nothing cleared.[/dim]")
            raise typer.Exit(code=1)

        await provider.open()
        try:
            for t in sorted(tables):
                count = await provider.clear(t)
                console.print(f"  {t}: {count} row(s) cleared (recoverable via `arc psqldb trash`)")
        finally:
            await provider.close()

    with _friendly_errors():
        asyncio.run(_run())
    console.print("[bold green]Clear complete.[/bold green]")


# ------------------------------------------------------------------------ #
# backup / restore — shell out to pg_dump/pg_restore, same "real client
# tool, not a reimplementation" philosophy as `connect` above. Scoped
# optionally to one plugin (every table it owns) or one table.
# ------------------------------------------------------------------------ #
@app.command()
def backup(
    plugin: str = typer.Option(None, "-p", "--plugin", help="Only back up this plugin's tables."),
    table: str = typer.Option(None, "-t", "--table", help="Only back up this one table."),
    out: Path = typer.Option(None, "--out", help="Output file. Defaults to backups/db/<timestamp>.dump."),
) -> None:
    """pg_dump the database (or a subset of it) into backups/db/."""
    root = _root_or_exit()
    if shutil.which("pg_dump") is None:
        err_console.print("`pg_dump` was not found on PATH. Install the PostgreSQL client tools.")
        raise typer.Exit(code=1)

    dsn = _dsn()
    if out is None:
        import datetime as dt
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        out = root / "backups" / "db" / f"{ts}.dump"
    out.parent.mkdir(parents=True, exist_ok=True)

    argv = ["pg_dump", "-Fc", "-f", str(out), dsn]
    tables = _tables_for_scope(root, plugin, table)
    for t in tables:
        argv += ["-t", t]

    console.print(f"[dim]$ {' '.join(argv[:-1])} <dsn>[/dim]")
    subprocess.run(argv, check=True)
    console.print(f"[bold green]Backup written to {out}[/bold green]")


@app.command()
def restore(
    dump_file: Path = typer.Argument(..., help="A file produced by `arc psqldb backup`."),
    plugin: str = typer.Option(None, "-p", "--plugin", help="Only restore this plugin's tables."),
    table: str = typer.Option(None, "-t", "--table", help="Only restore this one table."),
) -> None:
    """pg_restore a dump produced by `arc psqldb backup`."""
    root = _root_or_exit()
    if shutil.which("pg_restore") is None:
        err_console.print("`pg_restore` was not found on PATH. Install the PostgreSQL client tools.")
        raise typer.Exit(code=1)
    if not dump_file.exists():
        err_console.print(f"{dump_file} does not exist.")
        raise typer.Exit(code=1)

    dsn = _dsn()
    argv = ["pg_restore", "-d", dsn, "--clean", "--if-exists"]
    tables = _tables_for_scope(root, plugin, table)
    for t in tables:
        argv += ["-t", t]
    argv.append(str(dump_file))

    console.print(f"[dim]$ pg_restore -d <dsn> --clean --if-exists ... {dump_file}[/dim]")
    subprocess.run(argv, check=True)
    console.print("[bold green]Restore complete.[/bold green]")


def _tables_for_scope(root: Path, plugin: str | None, table: str | None) -> list[str]:
    if table:
        return [table]
    if plugin:
        provider = _boot()
        return [s.table for s in provider.schemas() if s.plugin == plugin]
    return []


# ------------------------------------------------------------------------ #
# trash — recover a document (or a dropped column's values) from _trash.
# ------------------------------------------------------------------------ #
@trash_app.command(name="list")
def trash_list(table: str = typer.Option(None, "-t", "--table")) -> None:
    """List un-recovered _trash entries, optionally scoped to one table."""
    dsn = _dsn()

    async def _run():
        conn = await asyncpg.connect(dsn)
        try:
            query = 'SELECT id, "table", drop_type, deleted_at FROM _trash WHERE restored_at IS NULL'
            params = []
            if table:
                query += ' AND "table" = $1'
                params.append(table)
            query += " ORDER BY deleted_at DESC LIMIT 50"
            return await conn.fetch(query, *params)
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    if not rows:
        console.print("[dim]_trash is empty.[/dim]")
        return
    for r in rows:
        console.print(f"{r['id']}  {r['table']:<20} {r['drop_type']:<8} deleted {r['deleted_at']}")


@trash_app.command(name="recover")
def trash_recover(trash_id: str = typer.Argument(..., help="The _trash row's own id (not the original document's).")) -> None:
    """Recover a _trash entry: Row/Table entries are re-inserted as-is;
    Column entries update just that one column back onto its original row."""
    dsn = _dsn()

    async def _run():
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow("SELECT * FROM _trash WHERE id = $1", trash_id)
            if row is None:
                err_console.print(f"No _trash entry with id '{trash_id}'.")
                raise typer.Exit(code=1)
            if row["restored_at"] is not None:
                err_console.print(f"_trash entry '{trash_id}' was already restored at {row['restored_at']}.")
                raise typer.Exit(code=1)

            import json
            raw_snapshot = row["snapshot"]
            snapshot_text = raw_snapshot if isinstance(raw_snapshot, str) else json.dumps(raw_snapshot)
            table = row["table"]

            # jsonb_populate_record does server-side, per-column type
            # coercion (timestamps, uuids, ints, ...) — this connection has
            # no custom codecs registered (unlike arc.psqldb's own pool), so
            # the snapshot's ISO-formatted strings etc. would otherwise fail
            # to bind as native Python types on the client side.
            if row["drop_type"] in ("Row", "Table"):
                await conn.execute(
                    f'INSERT INTO "{table}" '
                    f'SELECT * FROM jsonb_populate_record(null::"{table}", $1::jsonb) '
                    f'ON CONFLICT (id) DO NOTHING',
                    snapshot_text,
                )
            else:  # Column
                snapshot = json.loads(snapshot_text)
                row_id = snapshot["_row_id"]
                col = next(k for k in snapshot if k != "_row_id")
                await conn.execute(
                    f'UPDATE "{table}" t SET "{col}" = x."{col}" '
                    f'FROM jsonb_populate_record(null::"{table}", $1::jsonb) AS x '
                    f'WHERE t.id = $2',
                    snapshot_text, row_id,
                )

            await conn.execute("UPDATE _trash SET restored_at = now() WHERE id = $1", trash_id)
        finally:
            await conn.close()

    asyncio.run(_run())
    console.print(f"[bold green]Recovered _trash entry {trash_id}.[/bold green]")