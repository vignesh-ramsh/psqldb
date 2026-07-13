"""
psqldb.model
-------------------
Loads one plugin's `schemas/*.json` directory into TableSchema objects.

Convention (business-plugin side): a plugin declares its own tables under
`plugins/<name>/schemas/<Table Name>.json` — one file per table, filename
(slugified) becomes the physical table name. A business plugin's register()
calls `arc.psqldb.register_model(path_to_its_schemas_dir)` — see
psqldb.migrate.ModelRegistry — which is how psqldb discovers what to diff
and apply, without ever importing a business plugin's Python code to do it.

System-field injection (never declared by a business plugin):
  * normal table:  id, created_at, updated_at, created_by, updated_by, _state
  * child table:   id, parent, idx, created_at, updated_at, created_by,
                    updated_by, _state — each child row tracks its own
                    creation/update independently of the parent row; there
                    is no inheritance from the parent's own audit fields.
  * `"system": true` table: none of the above — the file declares every
    field itself (used for _trash / _field_registry / _audit_* only).

`_state` exists on every table (child tables included) because the soft-
delete trigger (psqldb.ddl.SOFT_DELETE_TRIGGER_SQL) cascades a child's
`_state` to 99 the same way it does the parent's — it needs the column to
exist everywhere it might fire.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from .fields import Field, FieldError, parse_field

_SLUG_RE = re.compile(r"[^a-z0-9_]+")
POSTGRES_IDENTIFIER_LIMIT = 63  # NAMEDATALEN - 1; Postgres silently truncates past this


class SchemaError(ValueError):
    """A schema file itself (not one field in it) is invalid."""


def slugify_table_name(filename_stem: str) -> str:
    """"Legal Tasks" -> "legal_tasks". Must produce a valid, boring Postgres
    identifier — collisions and invalid names are rejected loudly at plan/
    migrate time, never silently mangled further than this one deterministic
    rule. Reserved system tables (_trash, _field_registry, _patch_history,
    _audit_*) can never collide with a business table name produced here:
    leading underscores are always stripped before the "starts with a
    letter" check, so this function can never itself produce one."""
    slug = _SLUG_RE.sub("_", filename_stem.strip().lower()).strip("_")
    if not slug or not slug[0].isalpha():
        raise SchemaError(
            f"schema filename '{filename_stem}' does not produce a valid "
            f"table name ('{slug}') — must start with a letter after "
            f"lowercasing and replacing non [a-z0-9_] characters with '_'."
        )
    if len(slug) > POSTGRES_IDENTIFIER_LIMIT:
        raise SchemaError(
            f"schema filename '{filename_stem}' produces a table name "
            f"('{slug}') longer than Postgres's {POSTGRES_IDENTIFIER_LIMIT}-"
            f"character identifier limit — Postgres would silently truncate "
            f"it, risking a collision with an unrelated table. Shorten the "
            f"filename."
        )
    return slug


NORMAL_SYSTEM_FIELDS = [
    Field(id="_id", name="id", type="REFERENCE_PK"),  # rendered specially, see ddl.py
    Field(id="_created_at", name="created_at", type="DATETIME"),
    Field(id="_updated_at", name="updated_at", type="DATETIME"),
    Field(id="_created_by", name="created_by", type="REFERENCE_UUID"),  # no FK target - psqldb doesn't know "users"
    Field(id="_updated_by", name="updated_by", type="REFERENCE_UUID"),
    Field(id="_state", name="_state", type="INT"),
]

CHILD_SYSTEM_FIELDS = [
    Field(id="_id", name="id", type="REFERENCE_PK"),
    Field(id="_parent", name="parent", type="REFERENCE_UUID"),  # FK target filled in once the owning parent is known
    Field(id="_idx", name="idx", type="INT"),
    Field(id="_created_at", name="created_at", type="DATETIME"),
    Field(id="_updated_at", name="updated_at", type="DATETIME"),
    # created_by/updated_by track who created/last-touched THIS child row
    # specifically — independent of the parent row's own created_by/updated_by,
    # exactly the same per-row semantics NORMAL_SYSTEM_FIELDS already has above.
    # No inheritance from the parent happens anywhere; a child row's audit
    # fields have only ever meant "this row," parent included.
    Field(id="_created_by", name="created_by", type="REFERENCE_UUID"),
    Field(id="_updated_by", name="updated_by", type="REFERENCE_UUID"),
    Field(id="_state", name="_state", type="INT"),
]


@dataclass(frozen=True)
class TableSchema:
    table: str                 # slugified physical table name
    plugin: str                # owning plugin name (attributed by the caller, not this file)
    source_path: Path
    system: bool
    audit: bool
    child: bool
    fields: list[Field]
    indexes: list[dict]
    system_fields: list[Field]  # already resolved for this table (empty if system=True or is_patch)
    is_patch: bool = False     # from plugins/<plugin>/patches/, not plugins/<plugin>/schemas/ —
                                # see psqldb.migrate: never creates its table (skip+warn if missing),
                                # diffed against ONLY the fields this same plugin already owns on that
                                # table (never another plugin's, and never TABLE-typed/child fields)

    def all_fields(self) -> list[Field]:
        return [*self.system_fields, *self.fields]

    def column_fields(self) -> list[Field]:
        return [f for f in self.all_fields() if f.is_column()]

    def child_fields(self) -> list[Field]:
        """This table's own TABLE-typed fields — each names a child schema
        this table owns (see resolve_child_owners in migrate.py)."""
        return [f for f in self.fields if f.type == "TABLE"]


def _parse_fields_and_indexes(
    raw: dict, path: Path, *, table: str, known_system_columns: set[str]
) -> tuple[list[Field], list[dict]]:
    raw_fields = raw.get("fields", [])
    if not isinstance(raw_fields, list):
        raise SchemaError(f"{path}: 'fields' must be a list.")

    seen_ids: dict[str, int] = {}
    seen_names: dict[str, int] = {}
    fields: list[Field] = []
    for i, rf in enumerate(raw_fields):
        try:
            f = parse_field(rf, table=table, index=i)
        except FieldError as exc:
            raise SchemaError(f"{path}: {exc}") from exc
        if f.id in seen_ids:
            raise SchemaError(
                f"{path}: field id '{f.id}' used twice (field #{seen_ids[f.id] + 1} "
                f"and #{i + 1}) — ids must be unique within a schema file."
            )
        if f.name in seen_names:
            raise SchemaError(
                f"{path}: field name '{f.name}' used twice (field #{seen_names[f.name] + 1} "
                f"and #{i + 1})."
            )
        seen_ids[f.id] = i
        seen_names[f.name] = i
        fields.append(f)

    raw_indexes = raw.get("index", [])
    if not isinstance(raw_indexes, list):
        raise SchemaError(f"{path}: 'index' must be a list.")
    indexes = []
    known_columns = {f.name for f in fields} | known_system_columns
    for idx in raw_indexes:
        key = idx.get("key")
        idx_fields = idx.get("fields")
        if not key or not isinstance(idx_fields, list) or not idx_fields:
            raise SchemaError(f"{path}: each 'index' entry needs a 'key' and a non-empty 'fields' list.")
        unknown = [f for f in idx_fields if f not in known_columns]
        if unknown:
            raise SchemaError(f"{path}: index '{key}' references unknown field(s) {unknown}.")
        indexes.append({"key": key, "fields": idx_fields})

    return fields, indexes


def load_schema_file(path: Path, *, plugin: str) -> TableSchema:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc}).") from exc
    if not isinstance(raw, dict):
        raise SchemaError(f"{path}: schema file must be a JSON object.")

    table = slugify_table_name(path.stem)
    system = bool(raw.get("system", False))
    audit = bool(raw.get("audit", False))
    child = bool(raw.get("child", False))

    if system and child:
        raise SchemaError(f"{path}: 'system' and 'child' cannot both be true.")

    system_fields = [] if system else (list(CHILD_SYSTEM_FIELDS) if child else list(NORMAL_SYSTEM_FIELDS))
    fields, indexes = _parse_fields_and_indexes(
        raw, path, table=table, known_system_columns={sf.name for sf in system_fields}
    )

    # Every normal (non-system, non-child) table must declare at least one
    # business unique field of its own. The auto-injected `id` doesn't count
    # — it's a surrogate key for framework use (FKs, soft-delete/_trash
    # bookkeeping), never meant to double as a table's real-world identity.
    # This forces every business table to carry a genuine natural key from
    # day one, which the Query Engine's reference-by-natural-key resolution
    # (docs/arc.MD) depends on. System tables declare their own structure
    # deliberately (_trash, _field_registry, ...) and child tables are
    # identified by (parent, idx), not a key of their own — both exempt.
    if not system and not child and not any(f.unique for f in fields):
        raise SchemaError(
            f"{path}: table '{table}' declares no field with \"unique\": true. "
            f"Every normal table needs at least one business-declared unique "
            f"field of its own — the framework's auto-generated 'id' doesn't "
            f"count. Mark whichever field is this table's natural key (e.g. "
            f"an employee_code, an order number, a slug) as \"unique\": true."
        )

    return TableSchema(
        table=table, plugin=plugin, source_path=path, system=system, audit=audit,
        child=child, fields=fields, indexes=indexes, system_fields=system_fields,
    )


def load_schemas_dir(schemas_dir: Path, *, plugin: str) -> list[TableSchema]:
    if not schemas_dir.exists():
        return []
    schemas = []
    for path in sorted(schemas_dir.glob("*.json")):
        schemas.append(load_schema_file(path, plugin=plugin))
    tables_seen: dict[str, Path] = {}
    for s in schemas:
        if s.table in tables_seen:
            raise SchemaError(
                f"plugin '{plugin}': schema files '{tables_seen[s.table]}' and "
                f"'{s.source_path}' both produce table name '{s.table}'."
            )
        tables_seen[s.table] = s.source_path
    return schemas


# ------------------------------------------------------------------------ #
# Patches — plugins/<plugin>/patches/<Table Name>.json. Same file shape as
# a schema minus "system"/"audit"/"child" (none apply — a patch never
# declares a new table, only adds fields to / modifies fields it already
# owns on one that exists). See psqldb.migrate for how these are diffed:
# ownership-scoped to this same plugin, skip-with-warning if the target
# table doesn't exist yet, and — unlike schemas — TABLE-typed (child-table)
# fields are rejected outright, since a patch can't create a table at all.
# ------------------------------------------------------------------------ #
def load_patch_file(path: Path, *, plugin: str) -> TableSchema:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc}).") from exc
    if not isinstance(raw, dict):
        raise SchemaError(f"{path}: patch file must be a JSON object.")

    table = slugify_table_name(path.stem)
    fields, indexes = _parse_fields_and_indexes(raw, path, table=table, known_system_columns=set())

    for f in fields:
        if f.type == "TABLE":
            raise SchemaError(
                f"{path}: field '{f.name}' is type TABLE — patches cannot declare "
                f"child tables (that requires \"child\": true in a schema, not a patch)."
            )

    return TableSchema(
        table=table, plugin=plugin, source_path=path, system=False, audit=False,
        child=False, fields=fields, indexes=indexes, system_fields=[], is_patch=True,
    )


def load_patches_dir(patches_dir: Path, *, plugin: str) -> list[TableSchema]:
    if not patches_dir.exists():
        return []
    patches = []
    for path in sorted(patches_dir.glob("*.json")):
        patches.append(load_patch_file(path, plugin=plugin))
    tables_seen: dict[str, Path] = {}
    for p in patches:
        if p.table in tables_seen:
            raise SchemaError(
                f"plugin '{plugin}': patch files '{tables_seen[p.table]}' and "
                f"'{p.source_path}' both target table '{p.table}'."
            )
        tables_seen[p.table] = p.source_path
    return patches