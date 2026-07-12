"""
psqldb.fields
-------------------
Canonical field types for a psqldb schema file. One type = one Postgres
column shape, plus which tier enforces it:

  * DB tier    — a real Postgres type/constraint (STRING, INT, REFERENCE, ...).
  * app tier   — no DB guarantee; checked in psqldb.validation before every
                 write (EMAIL, PHONE, SELECT). SELECT is deliberately NOT a
                 Postgres ENUM or CHECK constraint — options are a business
                 rule, not a storage-engine rule, and the project's own call
                 was to keep that in the validation layer so options can
                 change without a migration.

This module is Postgres-shaped on purpose (docs/arc.MD §3.4 addendum: psqldb
+ relay are one opinionated pair, not a generic multi-backend abstraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# "data" was shorthand used in early schema drafts for STRING; accepted as a
# permanent alias so nothing written that way breaks.
TYPE_ALIASES = {"DATA": "STRING"}

# canonical type -> Postgres type template. TABLE has no column shape at all
# — it's handled entirely in psqldb.model (a TABLE field generates a whole
# separate child table, not a column on this one).
_SQL_TEMPLATES: dict[str, str] = {
    "STRING": "VARCHAR({length})",
    "TEXT": "TEXT",
    "BOOLEAN": "BOOLEAN",
    "INT": "INTEGER",
    "FLOAT": "REAL",
    "DECIMAL": "NUMERIC({precision},{scale})",
    "DATE": "DATE",
    "TIME": "TIME",
    "DATETIME": "TIMESTAMPTZ",
    "JSON": "JSONB",
    "EMAIL": "VARCHAR({length})",
    "PHONE": "VARCHAR({length})",
    "SELECT": "VARCHAR({length})",
    "REFERENCE": "UUID",
}

RELATIONAL_TYPES = frozenset({"REFERENCE", "TABLE"})
APP_VALIDATED_TYPES = frozenset({"EMAIL", "PHONE", "SELECT", "TABLE"})
CANONICAL_TYPES = frozenset(_SQL_TEMPLATES) | {"TABLE"}

DEFAULT_STRING_LENGTH = 255


class FieldError(ValueError):
    """A field declaration in a schema file is invalid — always a hard
    `arc psqldb plan`/`migrate` failure, never silently coerced."""


def normalize_type(raw: str, *, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise FieldError(f"{where}: 'type' must be a non-empty string.")
    upper = raw.strip().upper()
    upper = TYPE_ALIASES.get(upper, upper)
    if upper not in CANONICAL_TYPES:
        raise FieldError(
            f"{where}: unknown type '{raw}' — expected one of "
            f"{sorted(CANONICAL_TYPES)} (aliases: {TYPE_ALIASES})."
        )
    return upper


@dataclass(frozen=True)
class Field:
    """One declared field, already normalized from raw schema-file JSON.

    `id` is the stable identity (e.g. "AA01") — diffing and rename-detection
    key off THIS, never off `name`. Renaming a field in a schema file (same
    id, new name) becomes a non-destructive RENAME COLUMN; changing the id
    is indistinguishable from dropping one field and adding another, on
    purpose — that's the escape hatch for "this really is a different field".
    """
    id: str
    name: str
    type: str
    required: bool = False
    unique: bool = False
    length: int | None = None
    precision: int | None = None
    scale: int | None = None
    default: Any = None
    options: tuple[str, ...] | None = None  # SELECT only — a fixed choice list
    target: str | None = None      # REFERENCE / TABLE only — table this points to
    target_field: str | None = None  # REFERENCE only — which field on `target` this
                                      # points at. None = the implicit PK ("id") — the
                                      # only option TABLE fields ever have. When set,
                                      # this field's REAL physical column type is the
                                      # TARGET field's type, not the plain "UUID" this
                                      # class's own sql_type() below always returns for
                                      # REFERENCE — that resolution needs every plugin's
                                      # schemas at once, so it can't happen on a single
                                      # Field in isolation; see psqldb.migrate.resolve_ref_columns.

    def is_column(self) -> bool:
        """False only for TABLE — it generates a child table, not a column
        on this one, so it never appears in this table's own DDL."""
        return self.type != "TABLE"

    def sql_type(self) -> str:
        """The type THIS field alone can determine. For a REFERENCE field
        with target_field set, this is deliberately NOT the field's real
        physical column type (that's resolved cross-schema, see
        target_field's docstring above) — callers that care about the real
        type for such a field must go through
        psqldb.migrate.resolve_ref_columns instead of calling this directly."""
        if not self.is_column():
            raise FieldError(f"field '{self.name}' (TABLE) has no column type.")
        template = _SQL_TEMPLATES[self.type]
        if self.type == "STRING" or self.type in ("EMAIL", "PHONE", "SELECT"):
            return template.format(length=self.length or DEFAULT_STRING_LENGTH)
        if self.type == "DECIMAL":
            if not self.precision:
                raise FieldError(
                    f"field '{self.name}' (DECIMAL) requires 'precision' "
                    f"(and optionally 'scale', default 0)."
                )
            return template.format(precision=self.precision, scale=self.scale or 0)
        return template

    def options_list(self) -> list[str]:
        if self.type != "SELECT":
            return []
        return list(self.options or ())


def parse_field(raw: dict, *, table: str, index: int) -> Field:
    """Parse+validate one raw field dict from a schema file into a Field."""
    where = f"schema '{table}', field #{index + 1}"

    field_id = raw.get("id")
    if not field_id or not isinstance(field_id, str):
        raise FieldError(f"{where}: missing required string key 'id' (a stable field identity, e.g. \"AA01\").")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise FieldError(f"{where} ('{field_id}'): missing required string key 'name'.")

    where = f"schema '{table}', field '{name}' ({field_id})"
    ftype = normalize_type(raw.get("type", ""), where=where)

    required = bool(raw.get("required", raw.get("req", False)))
    unique = bool(raw.get("unique", False))
    length = raw.get("length")
    precision = raw.get("precision")
    scale = raw.get("scale")
    default = raw.get("default")
    options = raw.get("options")
    target = raw.get("target")
    target_field = raw.get("target_field")

    if ftype == "DECIMAL" and precision is None:
        raise FieldError(f"{where}: type DECIMAL requires 'precision'.")
    if ftype == "SELECT":
        if not isinstance(options, list) or not options or not all(isinstance(o, str) and o for o in options):
            raise FieldError(f"{where}: type SELECT requires 'options' as a non-empty JSON array of strings.")
        options = tuple(options)
    elif options is not None:
        raise FieldError(f"{where}: 'options' is only valid for type SELECT.")
    if ftype in RELATIONAL_TYPES and not target:
        raise FieldError(f"{where}: type {ftype} requires 'target' (the table/child-schema it points to).")
    if target_field is not None:
        if ftype != "REFERENCE":
            raise FieldError(
                f"{where}: 'target_field' is only valid for type REFERENCE (got {ftype}) "
                f"— a TABLE field always links via the implicit parent/id relationship."
            )
        if not isinstance(target_field, str) or not target_field.strip():
            raise FieldError(f"{where}: 'target_field', when given, must be a non-empty string.")
        target_field = target_field.strip()
        if target_field == "id":
            target_field = None  # "id" IS the default — normalize away the redundancy
                                  # so every other file only ever has to handle one case
                                  # ("None means id"), not two spellings of it.

    return Field(
        id=field_id, name=name, type=ftype, required=required, unique=unique,
        length=length, precision=precision, scale=scale, default=default,
        options=options, target=target, target_field=target_field,
    )
