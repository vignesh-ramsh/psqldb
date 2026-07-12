"""
psqldb.validation
-------------------
App-tier field validation — for the types that have no DB-level guarantee
(fields.APP_VALIDATED_TYPES): EMAIL, PHONE, SELECT, TABLE. REFERENCE needs
no entry here — it's a real FK constraint, the DB already enforces it.

Run from PsqlDbProvider.insert()/update() (see psqldb/__init__.py) so real
CRUD is exercised now, ahead of Relay; Relay's future CRUD reuses this
module rather than re-implementing per-type checks.
"""

from __future__ import annotations

import re
from typing import Any

from .model import TableSchema

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[0-9\s\-()]{6,20}$")


class ValidationError(ValueError):
    pass


def validate_row(schema: TableSchema, data: dict[str, Any]) -> None:
    """Validates the app-tier fields of one row of `data` against `schema`.
    Raises ValidationError with every problem found (not just the first),
    so a caller can show them all at once instead of one-at-a-time."""
    problems: list[str] = []

    for f in schema.fields:
        if f.type not in ("EMAIL", "PHONE", "SELECT"):
            continue
        value = data.get(f.name)
        if value is None:
            continue
        if f.type == "EMAIL" and not _EMAIL_RE.match(str(value)):
            problems.append(f"'{f.name}': '{value}' is not a valid email address.")
        elif f.type == "PHONE" and not _PHONE_RE.match(str(value)):
            problems.append(f"'{f.name}': '{value}' is not a valid phone number.")
        elif f.type == "SELECT":
            options = f.options_list()
            if value not in options:
                problems.append(f"'{f.name}': '{value}' is not one of {options}.")

    if problems:
        raise ValidationError(f"validation failed for '{schema.table}': " + "; ".join(problems))


async def validate_references_exist(conn: Any, schema: TableSchema, data: dict[str, Any], ref_targets: dict[str, str]) -> None:
    """REFERENCE fields already have a DB-level FK, so this is defense in
    depth (a clearer error before the FK violation, not instead of it).

    Checks against `f.target_field or "id"` directly — no cross-schema
    resolution (psqldb.migrate.resolve_ref_columns) needed here, unlike DDL
    rendering: by the time a row is being inserted/updated, target_field has
    already been validated (at plan/migrate time) as a real, unique column
    on the target, so this only needs the field itself to know which column
    to check."""
    problems: list[str] = []
    for f in schema.fields:
        if f.type != "REFERENCE":
            continue
        value = data.get(f.name)
        if value is None:
            continue
        target_table = ref_targets.get(f.target)
        if not target_table:
            continue
        target_column = f.target_field or "id"
        exists = await conn.fetchval(
            f'select exists(select 1 from "{target_table}" where "{target_column}" = $1)', value
        )
        if not exists:
            problems.append(
                f"'{f.name}': referenced value '{value}' does not exist in "
                f"'{target_table}'.\"{target_column}\"."
            )
    if problems:
        raise ValidationError(f"validation failed for '{schema.table}': " + "; ".join(problems))
