"""
psqldb.schema_spec
-------------------
Typed description of a schema/patch file's shape, expressed once as
arc.codec Structs (msgspec) instead of only by hand in psqldb.fields /
psqldb.model. Two things fall out of this ONE description, for free,
entirely locally (arc.codec is pure Python/msgspec introspection — no
network access, ever, at any point):

  * a real JSON Schema document (schema_json() / patch_schema_json()) an
    editor can point at for autocomplete and inline validation of every
    schemas/*.json and patches/*.json file in the project;
  * a fast structural pre-check (validate_schema_dict / validate_patch_dict)
    that runs before psqldb.model's own business-rule checks — catches
    "wrong JSON shape" mistakes (a typo'd key, an unknown type name, a
    string where a list belongs) with one clear message, before any of the
    deeper field-by-field parsing in psqldb.fields.parse_field even starts.

This does NOT replace psqldb.model / psqldb.fields's own checks —
cross-field and cross-file business rules ("a table needs exactly one
unique field", "primary_key is UUID-only", "options is SELECT-only",
"field ids must be unique within a file", ...) stay exactly where they
already are. This is a stricter, faster first pass in front of them, not a
second implementation of them.

Two small, deliberate tightenings over today's hand-rolled parsing (both
low-risk — verified against every schema/patch file already in this repo):
  * `type` must be spelled in canonical UPPERCASE (psqldb.fields.normalize_type
    is case-insensitive; every real file already uses uppercase, so this
    changes nothing in practice, only tightens what a *new* file must do to
    validate).
  * a patch file may not declare "system"/"audit"/"child" at all — those
    keys never apply to a patch (a patch never creates a table), and
    accepting-but-ignoring them today most likely means a copy-paste
    mistake from a schema file went unnoticed.
"""

from __future__ import annotations

from typing import Any, Literal

import msgspec

from arc.codec import CodecError, Struct
from arc.codec import schema as codec_schema
from arc.codec import validate as codec_validate

from .fields import CANONICAL_TYPES, TYPE_ALIASES

# Every string psqldb.fields.normalize_type() would accept, built from the
# SAME constants it already uses — not hand-copied, so this can never
# quietly drift from what actually gets accepted at load time.
_TYPE_NAMES = tuple(sorted(CANONICAL_TYPES | set(TYPE_ALIASES)))
FieldType = Literal[_TYPE_NAMES]


class IndexSpec(Struct, forbid_unknown_fields=True):
    key: str
    fields: list[str]


class UniqueTogetherSpec(Struct, forbid_unknown_fields=True):
    """A composite (multi-column) uniqueness constraint — the natural-key
    shape a single field's own `"unique": true` can't express (e.g. one
    attendance row per employee PER DAY: neither `employee` nor `date`
    alone is unique, only the pair). Same {key, fields} shape as
    IndexSpec on purpose — same authoring convention, different meaning:
    an index is a performance hint (never auto-dropped, §3.9); this is a
    real correctness constraint, rendered as a genuine Postgres UNIQUE
    constraint and diffed like one (psqldb.migrate)."""

    key: str
    fields: list[str]


class FieldSpec(Struct, forbid_unknown_fields=True):
    id: str
    name: str
    type: FieldType
    required: bool = False
    req: bool = False  # legacy alias for `required` (psqldb.fields.parse_field
    # still reads raw.get("required", raw.get("req", False))) — kept here only
    # so an existing file spelled this way doesn't fail "unknown field" for
    # no reason; new files should use `required`.
    unique: bool = False
    primary_key: bool = False
    # NOT named `list` — a Struct field named `list` shadows the builtin
    # `list` type during msgspec's own (lazy, string-based, PEP 563) type
    # resolution for every OTHER field in this same class that uses
    # `list[...]` as its annotation (options, and IndexSpec.fields below),
    # breaking with a cryptic "'member_descriptor' object is not
    # subscriptable" — verified directly against every real schema file in
    # this repo, which is what surfaced this. msgspec.field(name=...) keeps
    # the JSON key spelled "list" without the Python attribute needing to be.
    list_view: bool = msgspec.field(name="list", default=True)
    length: int | None = None
    precision: int | None = None
    scale: int | None = None
    default: Any = None
    options: list[str] | None = None
    target: str | None = None
    target_field: str | None = None
    group: str | None = None  # UI-only Row Editor section hint — see Field.group (psqldb.fields)


class SchemaFileSpec(Struct, forbid_unknown_fields=True):
    """plugins/<plugin>/schemas/<Table Name>.json"""

    system: bool = False
    audit: bool = False
    child: bool = False
    fields: list[FieldSpec] = []
    index: list[IndexSpec] = []
    unique_together: list[UniqueTogetherSpec] = []


class PatchFileSpec(Struct, forbid_unknown_fields=True):
    """plugins/<plugin>/patches/<Table Name>.json — deliberately has no
    system/audit/child: a patch never creates a table, so those keys never
    apply here (see module docstring). `unique_together` is schema-only for
    the same reason — a patch never creates a table, so it can't declare
    the table's own natural key either; `forbid_unknown_fields=True` above
    already rejects it here with a clear structural error."""

    fields: list[FieldSpec] = []
    index: list[IndexSpec] = []


def schema_json() -> dict:
    """The full local JSON Schema for a schemas/*.json file."""
    return codec_schema(SchemaFileSpec)


def patch_schema_json() -> dict:
    """The full local JSON Schema for a patches/*.json file."""
    return codec_schema(PatchFileSpec)


def validate_schema_dict(raw: dict) -> None:
    """Structural pre-check for an already-JSON-parsed schema file. Raises
    arc.codec.CodecError naming exactly what's wrong; the caller (psqldb.model)
    wraps this into its own SchemaError with the file path attached."""
    codec_validate(raw, type=SchemaFileSpec)


def validate_patch_dict(raw: dict) -> None:
    """Same as validate_schema_dict, for a patches/*.json file."""
    codec_validate(raw, type=PatchFileSpec)


__all__ = [
    "CodecError",
    "FieldSpec",
    "FieldType",
    "IndexSpec",
    "PatchFileSpec",
    "SchemaFileSpec",
    "UniqueTogetherSpec",
    "patch_schema_json",
    "schema_json",
    "validate_patch_dict",
    "validate_schema_dict",
]
