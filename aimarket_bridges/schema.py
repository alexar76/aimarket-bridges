"""JSON Schema → pydantic model, for frameworks that will not take a schema as data.

The three target frameworks disagree about how a tool declares its arguments, and the
disagreement is the whole reason this module exists (verified against installed versions on
2026-07-29, not from memory — all three had moved past what documentation suggested):

    langchain-core 1.5.2   `args_schema` accepts a raw JSON Schema **dict**. Nothing to do.
    crewai 1.15.8          `args_schema` accepts a dict and converts it ITSELF — but its
                           converter cannot, so a model must be supplied. See below.
    autogen-core 0.7.5     `FunctionTool` derives the schema from a function's type
                           ANNOTATIONS; `BaseTool` takes an explicit `args_type` model.

An earlier version of this note claimed crewai raises ``AttributeError: 'dict' object has no
attribute 'model_fields'`` on a dict. That was wrong, and worth correcting rather than
quietly deleting: the conclusion happened to be right for a reason nobody had measured. What
1.15.8 actually does is feed the dict to its own ``create_model_from_schema``, which refuses
UNION TYPES — ``Unsupported JSON schema type: ['string', 'integer']``. Ten of the 47 live
capabilities die there (percola, fermat, ablation, landauer and fourier, each producer and
its verifier), and this module builds all ten. Even on the 37 that survive, crewai's
converter knows nothing of the alias inversion below, so the rewritten name would go out on
the wire. So: hand crewai a model, and not because a dict cannot be passed.

So a capability whose interface arrives as JSON — which is what a federated hub can serve,
since the capability lives on someone else's machine — needs a model built at runtime for
two of the three.

Scope is deliberately the schemas that actually exist rather than the specification. All 47
capabilities in the live catalogue were surveyed first: primitives, union types like
``["string", "integer"]``, arrays including arrays-of-arrays, nested objects, ``enum``,
``default``, ``minimum``/``maximum``, ``required`` — and ``oneOf``, which a first pass over
the catalogue missed because it only looked at top-level property keys. `unsupported_keywords`
walks into ``items`` too and found four capabilities using it: a fermat edge is either
``[from, to, cost]`` or an object, and a kantor point is either an array of numbers or a
single number. Both are genuine polymorphism, so ``oneOf``/``anyOf`` map to a Union.

That is exactly why the reporter exists: silently dropping a constraint lets a tool
advertise an interface it does not honour, and the failure then surfaces far away, as the
capability refusing input the model was told was valid. There is still no ``allOf`` or
``$ref``, so reference resolution is not implemented — and would be reported if it appeared.

The same trap applies to NAMES, which is why sanitised fields carry an ``alias``. Three live
properties are not legal Python identifiers — ``fourier.verify@v1``'s ``lambda`` (a keyword,
and REQUIRED) and the ``from`` key inside a ``fermat.route@v1`` / ``fermat.verify@v1`` edge
object. Renaming them to ``lambda_`` / ``from_`` and stopping there produces a tool that
advertises an argument no capability accepts and sends one no capability reads: a refusal on
a call that was already billed. The alias makes the rewrite round-trip, because pydantic
emits it in both ``model_json_schema()`` and ``model_dump(by_alias=True)``.
"""

from __future__ import annotations

import keyword
import re
from typing import Any, Literal, Union

from pydantic import ConfigDict, Field, create_model

__all__ = ["model_from_schema", "python_type_for", "unsupported_keywords", "SchemaError"]


class SchemaError(ValueError):
    """The schema cannot be represented as a pydantic model."""


# Keywords this module understands. Anything else in a schema is reported rather than
# quietly ignored — see unsupported_keywords().
_KNOWN_ROOT = frozenset(
    {"type", "properties", "required", "description", "title", "additionalProperties",
     "default", "examples"}
)
_KNOWN_PROP = frozenset(
    {"type", "description", "title", "default", "enum", "items", "properties", "required",
     "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
     "minLength", "maxLength", "pattern", "additionalProperties", "format", "examples",
     "oneOf", "anyOf"}
)
_UNMODELLED = frozenset({"allOf", "not", "$ref", "$defs", "definitions",
                         "if", "then", "else", "patternProperties", "propertyNames"})

_PRIMITIVES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
}

#: Largest ``enum`` rendered as a Literal. The biggest in the live catalogue has 4 members.
MAX_ENUM_MEMBERS = 64

_IDENT = re.compile(r"[^0-9a-zA-Z_]")


def _safe_name(raw: str) -> str:
    """A usable Python identifier for a model name, never a keyword."""
    name = _IDENT.sub("_", raw).strip("_") or "Args"
    if name[0].isdigit():
        name = f"C{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    return name


def python_type_for(spec: Any, *, path: str = "") -> Any:
    """Python type annotation for one JSON Schema property.

    ``Any`` for an untyped or unrecognised property: a tool that accepts a value it cannot
    describe is still usable, whereas refusing to build the whole model over one loose field
    would drop a working capability from the catalogue entirely.
    """
    if not isinstance(spec, dict):
        return Any

    # An enum pins the value set more tightly than its type does, and every framework here
    # forwards Literal into the JSON Schema it shows the model.
    enum = spec.get("enum")
    if isinstance(enum, list) and enum and all(
        isinstance(v, (str, int, float, bool)) for v in enum
    ):
        if len(enum) > MAX_ENUM_MEMBERS:
            # Every member is rendered into the JSON Schema the model is shown, so a
            # peer-authored 100k-member enum is a token bomb in the tool definition — paid for
            # on every single request, whether the tool is called or not. Falling back to the
            # member type keeps the tool usable and loses only the value list, which
            # `unsupported_keywords` reports so the looser contract is visible.
            base = type(enum[0])
            return base if base in (str, int, float, bool) else Any
        return Literal[tuple(enum)]  # type: ignore[return-value]

    # oneOf / anyOf: genuinely different accepted shapes. Treated identically here — the
    # distinction is that oneOf demands exactly one branch match and anyOf at least one,
    # which is a validation nuance no framework's tool-schema layer enforces anyway.
    branches = spec.get("oneOf") or spec.get("anyOf")
    if isinstance(branches, list) and branches:
        members = []
        for i, branch in enumerate(branches):
            member = python_type_for(branch, path=f"{path}_alt{i}")
            if member is Any:
                # One unrepresentable branch makes the whole union meaningless: Any already
                # admits everything the other branches would.
                return Any
            if member not in members:
                members.append(member)
        if not members:
            return Any
        if len(members) == 1:
            return members[0]
        return Union[tuple(members)]  # type: ignore[return-value]

    declared = spec.get("type")

    # Union types, e.g. {"type": ["string", "integer"]} — 8 of these in the live catalogue.
    if isinstance(declared, list):
        members = [_PRIMITIVES[t] for t in declared if t in _PRIMITIVES]
        # A union that names a container is not expressible as a simple Union of primitives.
        if any(t in ("array", "object") for t in declared):
            return Any
        if not members:
            return Any
        if len(members) == 1:
            return members[0]
        return Union[tuple(members)]  # type: ignore[return-value]

    if declared in _PRIMITIVES:
        return _PRIMITIVES[declared]

    if declared == "array":
        items = spec.get("items")
        if isinstance(items, list):
            # Tuple-style `items` (positional). Not in the live catalogue; a plain list
            # keeps the tool callable instead of failing to build.
            return list[Any]
        return list[python_type_for(items, path=f"{path}[]")] if items else list[Any]

    if declared == "object":
        nested = spec.get("properties")
        if isinstance(nested, dict) and nested:
            return model_from_schema(spec, name=_safe_name(path or "Nested"))
        return dict[str, Any]

    return Any


def _unique_field_name(prop: str, taken: set[str]) -> str:
    """A field name for ``prop`` that no earlier property in this model already claimed.

    ``_safe_name`` is not injective: ``a-b`` and ``a_b`` both become ``a_b``, as do ``x.y``
    and ``x_y``, and ``from`` collides with a literal ``from_``. Building the field dict
    without checking meant the second property simply overwrote the first, and the damage was
    not one lost optional argument:

        properties: {"a-b": required string, "a_b": optional integer}
        -> model fields:  ['a_b']
        -> schema shown to the model:  properties ['a_b'], required None

    The requirement vanished entirely, so the model never sends ``a-b``, the capability
    refuses the call, and the call was already billed. The surviving field's alias also
    carried the WRONG property name onto the wire.

    Suffixing is deterministic — the same schema must produce the same model on every run, or
    a saved agent graph stops matching the tools it was built against — and the alias keeps
    each field pointing at the property it actually came from, so both go out correctly.
    """
    base = _safe_name(prop)
    if base not in taken:
        taken.add(base)
        return base
    for i in range(2, 1000):
        candidate = f"{base}_{i}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    raise SchemaError(f"cannot find a free field name for property {prop!r}")


def _field_for(spec: Any, *, required: bool, alias: str | None = None) -> Any:
    """A pydantic FieldInfo carrying the description and default the schema declares."""
    description = spec.get("description") if isinstance(spec, dict) else None
    kwargs: dict[str, Any] = {}
    if description:
        # The one piece of metadata that matters most: it is what the calling model reads to
        # decide whether this tool is the right one and what to pass it.
        kwargs["description"] = str(description)
    if alias:
        # Only set when _safe_name had to rewrite the property name. The alias is the name the
        # CAPABILITY uses, and it is what pydantic then emits in `model_json_schema()`
        # (by_alias=True is its default) and in `model_dump(by_alias=True)` — so the model is
        # shown the real argument name and the invoke body carries it back. Without this the
        # rewrite is one-way: `lambda` is advertised and sent as `lambda_`, which is a
        # guaranteed refusal on a call the operator has already paid for.
        kwargs["alias"] = alias

    if required:
        return Field(..., **kwargs)
    if isinstance(spec, dict) and "default" in spec:
        return Field(spec["default"], **kwargs)
    return Field(None, **kwargs)


def model_from_schema(schema: dict[str, Any], *, name: str = "Args") -> type:
    """Build a pydantic model from a JSON Schema object.

    An optional field becomes ``T | None`` with a default rather than staying required:
    pydantic would otherwise demand every property, and a model asked to fill in arguments
    it was told are optional invents values for them.
    """
    if not isinstance(schema, dict):
        raise SchemaError(f"schema must be an object, got {type(schema).__name__}")

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        # A capability taking no arguments is legitimate (a beacon, a status read). An empty
        # model is the honest representation, not an error.
        return create_model(_safe_name(name))

    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()

    fields: dict[str, Any] = {}
    taken: set[str] = set()
    for prop, spec in properties.items():
        field_name = _unique_field_name(prop, taken)
        is_required = prop in required_names
        annotation = python_type_for(spec, path=f"{name}_{field_name}")
        if not is_required:
            annotation = annotation | None if annotation is not Any else Any
        # `alias` whenever the field name differs from the property the capability expects —
        # three live properties need it, and one of them (fourier.verify@v1's `lambda`) is
        # REQUIRED, so without the alias that capability cannot be called at all. `from` in a
        # fermat edge is the nested case. A de-collided name (see _unique_field_name) needs it
        # for the same reason.
        alias = prop if field_name != prop else None
        fields[field_name] = (annotation, _field_for(spec, required=is_required, alias=alias))

    # populate_by_name so BOTH spellings validate. Callers that were written against the
    # pydantic field name (crewai's bridge passes `lambda_=...`) keep working, while a model
    # reading the aliased JSON Schema can send the capability's own `lambda`.
    return create_model(
        _safe_name(name), __config__=ConfigDict(populate_by_name=True), **fields
    )


def unsupported_keywords(schema: dict[str, Any]) -> list[str]:
    """Schema keywords present but not modelled, as ``path:keyword`` strings.

    Exists so a bridge can WARN instead of pretending. A dropped ``oneOf`` or ``$ref``
    means the tool advertises an interface it does not actually honour, and the failure
    surfaces much later as the capability refusing input the model was told was valid.
    """
    found: list[str] = []

    def walk(node: Any, path: str, known: frozenset[str]) -> None:
        if not isinstance(node, dict):
            return
        enum = node.get("enum")
        if isinstance(enum, list) and len(enum) > MAX_ENUM_MEMBERS:
            found.append(f"{path or '.'}:enum[{len(enum)}>{MAX_ENUM_MEMBERS}]")
        for key in node:
            if key in _UNMODELLED:
                found.append(f"{path or '.'}:{key}")
            elif key not in known:
                found.append(f"{path or '.'}:{key}")
        for prop, spec in (node.get("properties") or {}).items():
            walk(spec, f"{path}.{prop}" if path else prop, _KNOWN_PROP)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{path}[]", _KNOWN_PROP)

    walk(schema, "", _KNOWN_ROOT)
    return sorted(set(found))
