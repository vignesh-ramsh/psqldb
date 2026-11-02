"""
pgdb.routines
-------------
Run-once Python routines — a plugin's one-time data/setup work (create the
roles it needs, seed reference rows, backfill a new column) that is code,
not schema. Schemas/patches stay declarative JSON; anything that needs
logic lives here instead.

    plugins/<name>/routines/001_add_roles.py

        async def run() -> None:
            await arc.relay.save("Role", {...})

Registered via `pgdb.register_routines(Path(...) / "routines")` in the
plugin's register(kernel), applied by `arc pgdb migrate` (after the schema
step) or on their own with `arc pgdb routines`. Never on boot, never on
`arc build` — a routine writes data, and data work must be something an
operator asked for.

Rules this module enforces:

  * One file = one routine, identified by (plugin, file stem). Files whose
    name starts with `_` are ignored. Order WITHIN a plugin is filename
    order (prefix with `001_`, `002_`, ... when one routine depends on
    another); order ACROSS plugins is boot order, i.e. `requires` order.
  * A routine is an `async def run()` with no arguments. A file with no
    top-level `run` (the scaffolded sample, with its example commented out)
    is inactive: never run, never recorded.
  * A routine runs once. After it succeeds, a row in `_routines_history`
    records it together with the file's sha256; a later edit to an
    already-applied file is reported (status "changed") but NOT re-run.
  * Each relay/pgdb call inside a routine commits on its own — there is no
    routine-wide transaction. A routine that fails part-way is recorded as
    NOT applied and will run again from the top, so a routine MUST be safe
    to run again (upsert / exists-check, never a bare insert).
  * The first failure stops the run: later routines stay pending.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import introspect

_logger = logging.getLogger("pgdb.routines")

RoutineState = Literal["ready", "inactive", "invalid"]
RoutineStatus = Literal["pending", "applied", "changed", "inactive", "invalid"]


class RoutineError(RuntimeError):
    """A routine could not be loaded or run (bad file, missing history table...)."""


class RoutineFailed(RoutineError):
    """A routine's run() raised. `routine` is the one that failed; the original
    exception is chained as __cause__."""

    def __init__(self, routine: Routine, cause: BaseException) -> None:
        super().__init__(
            f"Routine '{routine.plugin}/{routine.name}' failed: {type(cause).__name__}: {cause}"
        )
        self.routine = routine


@dataclass(frozen=True)
class Routine:
    plugin: str
    name: str  # file stem, e.g. "001_add_roles"
    path: Path
    checksum: str  # sha256 hex of the file's bytes
    state: RoutineState
    problem: str | None = None  # why state == "invalid"


@dataclass(frozen=True)
class RoutineEntry:
    routine: Routine
    status: RoutineStatus


def _inspect_file(plugin: str, path: Path) -> Routine:
    """Classifies one file WITHOUT importing it: parsed with ast so listing
    status never executes a plugin's code, and a file that is only comments
    (the scaffolded sample) is recognisably inactive."""
    source = path.read_bytes()
    checksum = hashlib.sha256(source).hexdigest()
    base = {"plugin": plugin, "name": path.stem, "path": path, "checksum": checksum}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return Routine(**base, state="invalid", problem=f"syntax error: {exc}")
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            return Routine(**base, state="ready")
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return Routine(**base, state="invalid", problem="run() must be `async def run()`")
    return Routine(**base, state="inactive")


def discover(dirs: list[tuple[str, Path]], *, plugin: str | None = None) -> list[Routine]:
    """Every routine file under the registered directories, in run order:
    `dirs` is already in boot order (register_routines is called from each
    plugin's register(), which the kernel runs in dependency order), and
    files within one directory sort by filename. A missing directory is
    fine — a plugin may register one it has not created yet."""
    found: list[Routine] = []
    for owner, directory in dirs:
        if plugin is not None and owner != plugin:
            continue
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            found.append(_inspect_file(owner, path))
    return found


async def applied_checksums(
    conn: Any, *, allow_missing: bool = False
) -> dict[tuple[str, str], str]:
    """(plugin, name) -> checksum recorded when it ran. If the history table
    does not exist yet: `allow_missing` returns {} (what `arc pgdb migrate`
    wants — on a fresh database its own schema step is about to create the
    table, and every routine is pending); otherwise raises RoutineError,
    since routines must never run against a database `arc pgdb migrate`
    has not bootstrapped."""
    if not await introspect.table_exists(conn, "_routines_history"):
        if allow_missing:
            return {}
        raise RoutineError(
            "_routines_history does not exist yet — run `arc pgdb migrate` first "
            "(it creates the table, then applies any pending routines)."
        )
    rows = await conn.fetch("SELECT plugin, name, checksum FROM _routines_history")
    return {(r["plugin"], r["name"]): r["checksum"] for r in rows}


def classify(routines: list[Routine], applied: dict[tuple[str, str], str]) -> list[RoutineEntry]:
    entries: list[RoutineEntry] = []
    for routine in routines:
        recorded = applied.get((routine.plugin, routine.name))
        status: RoutineStatus
        if recorded is not None:
            status = "applied" if recorded == routine.checksum else "changed"
        elif routine.state == "inactive":
            status = "inactive"
        elif routine.state == "invalid":
            status = "invalid"
        else:
            status = "pending"
        entries.append(RoutineEntry(routine, status))
    return entries


def pending(entries: list[RoutineEntry]) -> list[Routine]:
    """What a run would execute: pending, plus invalid ones (so a broken
    file stops the run loudly instead of being skipped in silence)."""
    return [e.routine for e in entries if e.status in ("pending", "invalid")]


def _load_run(routine: Routine) -> Callable[[], Awaitable[Any]]:
    if routine.state == "invalid":
        raise RoutineError(
            f"Routine '{routine.plugin}/{routine.name}' is invalid: {routine.problem}"
        )
    module_name = f"_arc_routine_{routine.plugin}_{routine.name}"
    spec = importlib.util.spec_from_file_location(module_name, routine.path)
    if spec is None or spec.loader is None:
        raise RoutineError(f"Cannot load routine file {routine.path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - re-raised as RoutineError with the file named
        sys.modules.pop(module_name, None)
        raise RoutineError(
            f"Routine '{routine.plugin}/{routine.name}' failed to import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    run = getattr(module, "run", None)
    if not inspect.iscoroutinefunction(run):
        raise RoutineError(f"Routine '{routine.plugin}/{routine.name}' has no `async def run()`.")
    return run


async def _record(conn: Any, routine: Routine, duration: float) -> None:
    await conn.execute(
        "INSERT INTO _routines_history (plugin, name, duration, checksum) "
        "VALUES ($1, $2, $3, $4) ON CONFLICT (plugin, name) DO NOTHING",
        routine.plugin,
        routine.name,
        duration,
        routine.checksum,
    )


async def run_pending(
    conn: Any,
    to_run: list[Routine],
    *,
    on_start: Callable[[Routine], None] | None = None,
    on_done: Callable[[Routine, float], None] | None = None,
) -> list[Routine]:
    """Runs `to_run` in order, recording each success in _routines_history
    on `conn` (the caller's migration-lock connection). Stops at the first
    failure by raising — routines before it stay recorded, it and the ones
    after it stay pending. Returns the routines that ran."""
    ran: list[Routine] = []
    for routine in to_run:
        run = _load_run(routine)
        if on_start is not None:
            on_start(routine)
        started = time.perf_counter()
        try:
            await run()
        except Exception as exc:  # noqa: BLE001 - re-raised as RoutineFailed, original chained
            _logger.exception("routine %s/%s failed", routine.plugin, routine.name)
            raise RoutineFailed(routine, exc) from exc
        duration = time.perf_counter() - started
        await _record(conn, routine, duration)
        ran.append(routine)
        if on_done is not None:
            on_done(routine, duration)
    return ran


async def mark_applied(conn: Any, routine: Routine) -> None:
    """Records `routine` as applied WITHOUT running it — for a database
    where its work was already done by hand."""
    await _record(conn, routine, 0.0)
