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


def validate_columns_known(schema: TableSchema, data: dict[str, Any]) -> None:
    """Every key in `data` must be a real column on `schema`. insert()/
    update() (psqldb/__init__.py) build SQL identifiers directly from these
    keys (`f'"{col}"'`) — an unknown key isn't just a typo worth a clear
    error, it's an unvalidated string about to be interpolated into a SQL
    identifier position. The Query Engine (relay/query.py) already
    whitelists every column this way for reads (parse_filters); this closes
    the matching gap on the write side."""
    known = {f.name for f in schema.column_fields()}
    unknown = [k for k in data if k not in known]
    if unknown:
        raise ValidationError(
            f"'{schema.table}': unknown field(s) {sorted(unknown)} — not a declared "
            f"column (known: {sorted(known)})."
        )


def friendly_fk_error(exc: Exception, *, table: str) -> ValidationError:
    """Translates asyncpg's ForeignKeyViolationError into the same
    ValidationError shape validate_row already raises. Replaces a separate
    pre-check SELECT per REFERENCE field that used to run before every
    insert/update (psqldb/__init__.py) — the DB already enforces this via
    the real FK constraint (ON DELETE RESTRICT, psqldb.ddl), so the
    pre-check was a redundant round-trip; catching the real violation
    instead costs nothing on the success path, which is the common case.

    Constraint names for a REFERENCE column follow one deterministic
    pattern (psqldb.ddl._user_column_sql's inline `REFERENCES` clause,
    never a named CONSTRAINT — Postgres auto-names it "{table}_{column}
    _fkey"), so this can usually name the offending field; falls back to
    Postgres's own `detail` message (always present) if the pattern
    doesn't match for some reason, rather than guessing."""
    constraint = getattr(exc, "constraint_name", "") or ""
    prefix, suffix = f"{table}_", "_fkey"
    field = constraint[len(prefix):-len(suffix)] if constraint.startswith(prefix) and constraint.endswith(suffix) else None
    detail = getattr(exc, "detail", None) or str(exc)
    if field:
        return ValidationError(f"'{table}': '{field}' references a row that doesn't exist ({detail})")
    return ValidationError(f"'{table}': {detail}")


def friendly_unique_error(exc: Exception, *, table: str) -> ValidationError:
    """Translates asyncpg's UniqueViolationError the same way
    friendly_fk_error (above) translates a FK violation — a `"unique": true`
    field's constraint follows Postgres's own default naming for a plain
    column UNIQUE constraint, "{table}_{column}_key" (confirmed against a
    real migrated table, e.g. "_users_email_key"), so this can usually name
    the offending field directly instead of surfacing a raw
    UniqueViolationError traceback for what's almost always an ordinary
    operator mistake (a duplicate email, a duplicate role name)."""
    constraint = getattr(exc, "constraint_name", "") or ""
    prefix, suffix = f"{table}_", "_key"
    field = constraint[len(prefix):-len(suffix)] if constraint.startswith(prefix) and constraint.endswith(suffix) else None
    detail = getattr(exc, "detail", None) or str(exc)
    if field:
        return ValidationError(f"'{table}': '{field}' must be unique ({detail})")
    return ValidationError(f"'{table}': {detail}")
