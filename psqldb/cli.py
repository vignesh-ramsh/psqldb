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
from urllib.parse import quote_plus, urlparse

import asyncpg
import typer
from rich.console import Console
from rich.markup import escape

import arc
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
def setup() -> None:
    """Interactively configure the Postgres connection (host/port/database/
    user/password), validating connectivity before saving — the guided
    alternative to hand-composing a DSN for `arc settings set psqldb_dsn
    ... --secret`. Safe to re-run: asks before overwriting an existing
    psqldb_dsn, showing what it would replace."""
    root = find_project_root()
    if root is None:
        err_console.print(
            "Not inside an ARC project (no .arc/arc.toml found here or in any parent)."
        )
        raise typer.Exit(code=1)

    mgr = SettingsManager(root / ".arc")
    existing = mgr.get(DSN_KEY, reveal=True)
    if existing is not None:
        parsed_existing = urlparse(existing)
        console.print(
            f"[yellow]psqldb_dsn is already set[/yellow] "
            f"({parsed_existing.hostname}:{parsed_existing.port or 5432}"
            f"/{parsed_existing.path.lstrip('/')})."
        )
        if not typer.confirm("Reconfigure it?", default=False):
            console.print("[dim]Aborted — nothing changed.[/dim]")
            raise typer.Exit(code=1)

    host = typer.prompt("Host", default="localhost")
    port = typer.prompt("Port", default=5432, type=int)
    database = typer.prompt("Database name")
    user = typer.prompt("User")
    # default="" (not omitted) — Click's prompt() re-prompts forever on
    # empty input when no default is set at all, which would make a
    # genuinely passwordless (trust-auth) local Postgres role impossible
    # to configure through this command.
    password = typer.prompt(
        "Password (blank for none)", default="", show_default=False, hide_input=True
    )

    # quote_plus so a special character in user/password (@, :, /, ...)
    # can't be misparsed as DSN structure by urlparse/asyncpg.
    auth = quote_plus(user)
    if password:
        auth += f":{quote_plus(password)}"
    dsn = f"postgresql://{auth}@{host}:{port}/{database}"

    async def _check() -> str:
        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            return await conn.fetchval("select version()")
        finally:
            await conn.close()

    console.print(f"Connecting to {host}:{port}/{database}...")
    try:
        version = asyncio.run(_check())
    except Exception as exc:
        err_console.print(f"FAILED to connect: {exc}")
        console.print("[dim]Nothing saved — run `arc psqldb setup` again to retry.[/dim]")
        raise typer.Exit(code=1)

    mgr.set(DSN_KEY, dsn, secret=True)
    console.print(f"[bold green]Connected[/bold green] — {version}")
    console.print(f"[dim]Saved to {DSN_KEY} (encrypted, .arc/arc.secrets).[/dim]")
    console.print(
        "[dim]Next: `arc psqldb migrate` to create tables in this database.[/dim]"
    )


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
            f"psqldb: FAILED to connect to {parsed.hostname}:{parsed.port or 5432} — {exc}"
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
        "-h",
        parsed.hostname or "localhost",
        "-p",
        str(parsed.port or 5432),
        "-U",
        parsed.username or "postgres",
        "-d",
        dbname,
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
        err_console.print(
            "Not inside an ARC project (no .arc/arc.toml found here or in any parent)."
        )
        raise typer.Exit(code=1)
    return root


def _build_plan(plugin: str | None, table: str | None) -> migrate.MigrationPlan:
    # The plugin-existence check needs boot()'s registered schemas/patches
    # but no live connection — done up front, sync, so a typo'd -p exits
    # before arc.runtime.run_async ever opens a pool for it.
    provider = _boot()
    if plugin and not any(s.plugin == plugin for s in [*provider.schemas(), *provider.patches()]):
        err_console.print(f"No schemas or patches registered for plugin '{plugin}'.")
        raise typer.Exit(code=1)

    async def _do() -> migrate.MigrationPlan:
        async with arc.psqldb.acquire() as conn:
            # Always diff the FULL schema set (a -p scope must not make other
            # plugins' tables look "no longer declared" to _dropped_table_ops),
            # then narrow both ops AND schemas to the scope — see _scope_plan.
            the_plan = await migrate.build_plan(
                conn, arc.psqldb.schemas(), arc.psqldb.patches(), only_table=table
            )
            if plugin:
                _scope_plan(the_plan, plugin)
            return the_plan

    return arc.runtime.run_async(_do(), open=("psqldb",))


def _scope_plan(plan: migrate.MigrationPlan, plugin: str) -> None:
    """Narrow a full plan to one plugin — BOTH the ops that will execute and
    the schemas whose _field_registry rows apply_plan will overwrite. The
    ops filter alone used to leave plan.schemas covering every plugin, so a
    scoped migrate recorded OTHER plugins' still-pending declarations in
    _field_registry as if their (filtered-out, never-executed) DDL had been
    applied — permanently hiding those pending changes from every later
    diff. Silent schema drift; both lists must stay in lockstep."""
    plan.ops = [op for op in plan.ops if op.plugin == plugin or op.table == "_bootstrap"]
    plan.schemas = [s for s in plan.schemas if s.plugin == plugin]


def _print_plan(plan: migrate.MigrationPlan) -> None:
    """Shows only the REAL diff (Op.maintenance ops — audit-table/index/
    trigger self-healing, appended unconditionally on every build_plan()
    call regardless of whether anything changed — are never displayed,
    though they still execute; see Op.maintenance's own docstring). A
    project with no genuine schema/patch change prints one line, not the
    always-present maintenance noise every previous call showed.

    Every piece of DYNAMIC text (table name, op description, the [safe]/
    [DESTRUCTIVE] tag itself, warnings) goes through rich.markup.escape()
    — found the hard way while writing this: Rich's markup parser treats
    ANY lowercase bracketed word as an attempted style name, and silently
    drops it (bracket text included) when that "style" doesn't resolve —
    which is exactly what "[safe]" is. `[DESTRUCTIVE]` (uppercase) only
    ever survived by coincidence, since Rich's own style names are never
    all-caps, so it doesn't match that lookalike pattern in the first
    place. Every non-destructive line in this command's entire history
    has been silently missing its own "[safe]" tag until this fix — and
    without escaping, a schema/patch file with a literal `[` in a field
    name or description would have the exact same problem, worse."""
    real_ops = plan.real_ops()
    if not real_ops:
        console.print("[dim]No schema changes to apply.[/dim]")
    else:
        by_table: dict[str, list[migrate.Op]] = {}
        for op in real_ops:
            by_table.setdefault(op.table, []).append(op)
        for table, ops in by_table.items():
            table_style = "bold red" if any(op.destructive for op in ops) else "bold green"
            console.print(f"[{table_style}]{escape(table)}[/{table_style}]")
            for op in ops:
                style = "bold red" if op.destructive else "green"
                tag = "DESTRUCTIVE" if op.destructive else "safe"
                console.print(
                    f"  [{style}]{escape(f'[{tag}]')}[/{style}] "
                    f"({escape(op.source)}) {escape(op.description)}"
                )
    for warning in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(warning)}")


SCHEMA_DIR_NAME = "schema"  # under .arc/ — where the generated local JSON Schema files live


@app.command()
def check(
    write_schema: bool = typer.Option(
        False,
        "--write-schema",
        help="Also (re)write the local JSON Schema files under .arc/schema/ and wire "
        ".vscode/settings.json to point every schemas/*.json and patches/*.json file at "
        "them, for editor autocomplete. Safe to re-run any time schema_spec.py changes.",
    ),
) -> None:
    """Validate every schemas/*.json and patches/*.json file across every
    plugin in the project — structural shape first (psqldb.schema_spec:
    unknown keys, wrong JSON types, an unrecognized `type` name — entirely
    local, no database, no network), then the existing business-rule
    checks (a unique field is required, primary_key is UUID-only, ...)
    layered on top exactly as they already run during `arc psqldb plan`/
    `migrate`. Unlike those commands, this never stops at the first bad
    plugin — it checks every plugin and reports every problem in one pass,
    so `arc psqldb plan` failing on file #1 doesn't hide a second, unrelated
    mistake in file #7."""
    root = _root_or_exit()
    from arc import registry

    from .model import SchemaError, load_patches_dir, load_schemas_dir

    manifests = registry.discover_plugins(root / "plugins")
    total = 0
    problems: list[tuple[str, str]] = []
    for m in manifests:
        for loader, subdir in ((load_schemas_dir, "schemas"), (load_patches_dir, "patches")):
            d = m.source_dir / subdir
            if not d.exists() or not any(d.glob("*.json")):
                continue
            try:
                items = loader(d, plugin=m.name)
            except SchemaError as exc:
                problems.append((m.name, str(exc)))
            else:
                total += len(items)

    if write_schema:
        _write_local_schemas(root)

    if problems:
        err_console.print(
            f"[bold red]{len(problems)} plugin(s) with problems[/bold red] (checked {total} file(s) OK elsewhere):"
        )
        for plugin_name, message in problems:
            err_console.print(f"  [bold]{plugin_name}[/bold]: {message}")
        raise typer.Exit(code=1)
    console.print(
        f"[bold green]OK[/bold green] — {total} schema/patch file(s) across "
        f"{len(manifests)} plugin(s), no problems found."
    )


def _write_local_schemas(root: Path) -> None:
    """Writes the two generated JSON Schema documents to .arc/schema/, and
    wires .vscode/settings.json's json.schemas to point every
    plugins/*/schemas/*.json and plugins/*/patches/*.json file at them —
    editor autocomplete, fully local (file paths, never a URL). Merges
    into an existing settings.json rather than overwriting it; safe to
    re-run any time schema_spec.py's Structs change."""
    import json as _json

    from . import schema_spec

    schema_dir = root / ".arc" / SCHEMA_DIR_NAME
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_path = schema_dir / "psqldb-schema.schema.json"
    patch_path = schema_dir / "psqldb-patch.schema.json"
    schema_path.write_text(_json.dumps(schema_spec.schema_json(), indent=2) + "\n")
    patch_path.write_text(_json.dumps(schema_spec.patch_schema_json(), indent=2) + "\n")
    console.print(f"[dim]wrote {schema_path}[/dim]")
    console.print(f"[dim]wrote {patch_path}[/dim]")

    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    settings_path = vscode_dir / "settings.json"
    try:
        settings = _json.loads(settings_path.read_text()) if settings_path.exists() else {}
    except _json.JSONDecodeError:
        err_console.print(
            f"[yellow]warning:[/yellow] {settings_path} is not valid JSON — leaving it "
            f"untouched. Add the json.schemas mapping to it by hand if you want editor "
            f"autocomplete (see .arc/schema/*.schema.json)."
        )
        return

    schemas_entry = {
        "fileMatch": ["plugins/*/*/schemas/*.json"],
        "url": "./.arc/schema/psqldb-schema.schema.json",
    }
    patches_entry = {
        "fileMatch": ["plugins/*/*/patches/*.json"],
        "url": "./.arc/schema/psqldb-patch.schema.json",
    }
    existing = settings.setdefault("json.schemas", [])
    for entry in (schemas_entry, patches_entry):
        if entry not in existing:
            existing.append(entry)
    settings_path.write_text(_json.dumps(settings, indent=2) + "\n")
    console.print(f"[dim]updated {settings_path}[/dim]")


@app.command()
def plan(
    plugin: str = typer.Option(
        None, "-p", "--plugin", help="Only show changes for this plugin's schemas/patches."
    ),
    table: str = typer.Option(None, "-t", "--table", help="Only show changes for this table."),
) -> None:
    """Preview what `arc psqldb migrate` would do — never touches the database."""
    _root_or_exit()
    with _friendly_errors():
        the_plan = _build_plan(plugin, table)
        _print_plan(the_plan)


@app.command()
def migrate_(
    plugin: str = typer.Option(
        None, "-p", "--plugin", help="Only migrate this plugin's schemas/patches."
    ),
    table: str = typer.Option(None, "-t", "--table", help="Only migrate this table."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (for CI/non-interactive use)."
    ),
) -> None:
    """Diff every registered schema AND patch against the live DB, show the
    plan, and — after confirmation — apply it. Always shows the plan first,
    even with --yes; destructive changes (drops, type/NOT NULL tightening)
    are always recoverable from _trash, but are still shown in red."""
    root = _root_or_exit()

    async def _do() -> bool:
        # boot() already ran (arc.runtime.run_async does it before opening
        # psqldb below) — arc.psqldb IS the same provider _boot() used to
        # return separately. Returns whether a REAL migration happened
        # (False for "nothing to do" or "maintenance ops only") — the
        # caller uses this to decide whether "Migration complete." is
        # worth printing at all.
        provider = arc.psqldb
        schemas, patches = provider.schemas(), provider.patches()
        target_plugins = {s.plugin for s in schemas if not plugin or s.plugin == plugin} | {
            p.plugin for p in patches if not plugin or p.plugin == plugin
        }
        reference = migrate.migration_reference()
        async with provider.acquire() as conn:
            # Held for the WHOLE diff-then-apply window, not just apply_plan
            # — two replicas racing must not both diff against the same
            # "before" state and then both try to apply overlapping DDL.
            # Blocks (with one logged notice, not a spinner-less hang) rather
            # than failing outright; see migration_lock's own docstring for
            # why this is safe to hold across the interactive confirm below.
            async with migrate.migration_lock(conn):
                the_plan = await migrate.build_plan(conn, schemas, patches, only_table=table)
                if plugin:
                    _scope_plan(the_plan, plugin)  # ops AND schemas — see _scope_plan
                _print_plan(the_plan)
                if not the_plan.has_real_changes():
                    if not the_plan.is_empty():
                        # Maintenance ops only (audit table/index/trigger
                        # self-healing, Op.maintenance's own docstring) —
                        # still applied, since they're genuinely idempotent
                        # and this IS what re-establishes something dropped
                        # by hand, just silently: no confirmation prompt,
                        # no per-plugin file below, nothing for a human to
                        # review, because nothing genuinely changed.
                        await migrate.apply_plan(conn, the_plan, reference=reference)
                    return False
                if not yes and not typer.confirm("Proceed?", default=False):
                    console.print("[dim]Aborted — nothing applied.[/dim]")
                    raise typer.Exit(code=1)
                await migrate.apply_plan(conn, the_plan, reference=reference)
        for plugin_name in sorted(target_plugins):
            plugin_ops_plan = migrate.MigrationPlan(
                ops=[op for op in the_plan.ops if op.plugin == plugin_name]
            )
            if plugin_ops_plan.has_real_changes():
                path = migrate.write_migration_file(
                    root / "plugins" / plugin_name, plugin_name, plugin_ops_plan, reference
                )
                console.print(f"[dim]wrote {path}[/dim]")
        return True

    with _friendly_errors():
        applied = arc.runtime.run_async(_do(), open=("psqldb",))
    if applied:
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
        console.print(
            f"{r['applied_at']}  {r['plugin']:<16} {r['table']:<20} {r['kind']:<6} {r['reference']}"
        )


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

    from .migrate import order_for_clear

    # Boot + compute the scope + confirm BEFORE opening a pool at all
    # (unchanged from before this used arc.runtime.run_async) — an aborted
    # confirmation should never have opened a connection in the first place.
    provider = _boot()
    tables = {table} if table else {s.table for s in provider.schemas() if s.plugin == plugin}
    if plugin and not tables:
        err_console.print(f"No schemas registered for plugin '{plugin}'.")
        raise typer.Exit(code=1)

    # Order matters: a table REFERENCING another (ON DELETE RESTRICT)
    # must be cleared first, or the soft-delete trigger's physical
    # DELETE on the referenced table hits a live FK violation the
    # moment some other row in this same clear still points at it.
    ordered = order_for_clear(tables, provider.ref_columns()) if len(tables) > 1 else sorted(tables)

    console.print(f"About to clear all rows from: {', '.join(ordered)}")
    if not yes and not typer.confirm("Proceed?", default=False):
        console.print("[dim]Aborted — nothing cleared.[/dim]")
        raise typer.Exit(code=1)

    async def _do() -> None:
        for t in ordered:
            count = await arc.psqldb.clear(t)
            console.print(f"  {t}: {count} row(s) cleared (recoverable via `arc psqldb trash`)")

    with _friendly_errors():
        arc.runtime.run_async(_do(), open=("psqldb",))
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
    out: Path = typer.Option(
        None, "--out", help="Output file. Defaults to backups/db/<timestamp>.dump."
    ),
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
        err_console.print(
            "`pg_restore` was not found on PATH. Install the PostgreSQL client tools."
        )
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
            query = (
                'SELECT id, "table", drop_type, deleted_at FROM _trash WHERE restored_at IS NULL'
            )
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
def trash_recover(
    trash_id: str = typer.Argument(
        ..., help="The _trash row's own id (not the original document's)."
    ),
) -> None:
    """Recover a _trash entry: Row/Table entries are re-inserted as-is;
    a Column entry's snapshot is a jsonb ARRAY covering up to
    psqldb.migrate.COLUMN_DROP_CHUNK_SIZE rows (one _trash entry = one
    chunk of the dropped column, not one row) — recovering it updates
    every row in that chunk back onto its original row in one set-based
    UPDATE, not just a single row."""
    dsn = _dsn()

    async def _run():
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow("SELECT * FROM _trash WHERE id = $1", trash_id)
            if row is None:
                err_console.print(f"No _trash entry with id '{trash_id}'.")
                raise typer.Exit(code=1)
            if row["restored_at"] is not None:
                err_console.print(
                    f"_trash entry '{trash_id}' was already restored at {row['restored_at']}."
                )
                raise typer.Exit(code=1)

            import json

            raw_snapshot = row["snapshot"]
            snapshot_text = (
                raw_snapshot if isinstance(raw_snapshot, str) else json.dumps(raw_snapshot)
            )
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
                    f"ON CONFLICT (id) DO NOTHING",
                    snapshot_text,
                )
            else:  # Column
                # snapshot is a jsonb ARRAY of {"_row_id": ..., "<col>": ...}
                # objects (one per row in this chunk) — jsonb_array_elements
                # unpacks it back into one row per element, and
                # jsonb_populate_record infers "col"'s real type from the
                # table's own (by-now re-added) column, same trick the
                # Row/Table branch above uses. _row_id has no matching
                # column on this table (the real PK is "id"), so it's
                # pulled out directly via ->> rather than through
                # jsonb_populate_record, which would just silently drop it.
                snapshot = json.loads(snapshot_text)
                col = next(k for k in snapshot[0] if k != "_row_id")
                await conn.execute(
                    f'UPDATE "{table}" t '
                    f'SET "{col}" = (jsonb_populate_record(null::"{table}", elem))."{col}" '
                    f"FROM jsonb_array_elements($1::jsonb) AS elem "
                    f"WHERE t.id = (elem->>'_row_id')::uuid",
                    snapshot_text,
                )

            await conn.execute("UPDATE _trash SET restored_at = now() WHERE id = $1", trash_id)
        finally:
            await conn.close()

    asyncio.run(_run())
    console.print(f"[bold green]Recovered _trash entry {trash_id}.[/bold green]")
