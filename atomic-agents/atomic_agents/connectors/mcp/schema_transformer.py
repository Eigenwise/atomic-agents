"""Module for transforming JSON schemas to Pydantic models."""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type, Tuple, Literal, Union, Annotated, cast

from atomic_agents.connectors.mcp.mcp_definition_service import MCPAttributeType
from jsonschema import Draft202012Validator, ValidationError as JSONSchemaValidationError, validators
from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError, WithJsonSchema, create_model
from pydantic_core import SchemaError, to_jsonable_python
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from atomic_agents.base.base_io_schema import BaseIOSchema

logger = logging.getLogger(__name__)


def _reject_false_schema(value: Any) -> Any:
    raise ValueError("Value is forbidden by a false JSON Schema")


_FalseSchemaValue = Annotated[Any, BeforeValidator(_reject_false_schema), WithJsonSchema({"not": {}})]


def _validate_ref(validator: Any, reference: str, value: Any, schema: Any) -> Any:
    try:
        yield from Draft202012Validator.VALIDATORS["$ref"](validator, reference, value, schema)
    except Unresolvable:
        # Match the transformer's Any fallback only for the unresolved reference.
        return


_ScalarChoiceValidator = validators.extend(Draft202012Validator, {"$ref": _validate_ref})


@lru_cache(maxsize=256)
def _pattern_adapter(pattern: str) -> Optional[TypeAdapter]:
    """Compile a pattern with Pydantic's regex engine, or None when the engine does not support it."""
    try:
        return TypeAdapter(Annotated[str, Field(pattern=pattern)])
    except SchemaError:
        return None


def _pattern_matches(pattern: Any, value: Any) -> Optional[bool]:
    """Search value with Pydantic's regex engine, or None when the pattern cannot be checked."""
    adapter = _pattern_adapter(pattern) if isinstance(pattern, str) and isinstance(value, str) else None
    if adapter is None:
        return None
    try:
        adapter.validate_python(value)
    except ValidationError:
        return False
    return True


# Server patterns must not reach Python's backtracking re; unsupported ones are skipped.
def _validate_pattern(validator: Any, pattern: Any, value: Any, schema: Any) -> Any:
    if _pattern_matches(pattern, value) is False:
        yield JSONSchemaValidationError(f"{value!r} does not match {pattern!r}")


def _validate_pattern_properties(validator: Any, pattern_properties: Any, value: Any, schema: Any) -> Any:
    if not validator.is_type(value, "object"):
        return
    for pattern, subschema in pattern_properties.items():
        for key, item in value.items():
            if _pattern_matches(pattern, key):
                yield from validator.descend(item, subschema, path=key, schema_path=pattern)


def _validate_additional_properties(validator: Any, additional: Any, value: Any, schema: Any) -> Any:
    if not validator.is_type(value, "object"):
        return
    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    # A key an unsupported pattern might cover is not counted as additional.
    extras = [
        key
        for key in value
        if key not in properties and all(_pattern_matches(pattern, key) is False for pattern in patterns)
    ]
    if validator.is_type(additional, "object"):
        for extra in extras:
            yield from validator.descend(value[extra], additional, path=extra)
    elif not additional and extras:
        yield JSONSchemaValidationError(f"Additional properties are not allowed ({', '.join(map(repr, extras))})")


_OneOfBranchValidator = validators.extend(
    _ScalarChoiceValidator,
    {
        "pattern": _validate_pattern,
        "patternProperties": _validate_pattern_properties,
        "additionalProperties": _validate_additional_properties,
    },
)


class _OneOfJsonSchema:
    """Advertise every converted branch with oneOf, where Pydantic's anyOf would merge duplicates."""

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Any) -> Dict[str, Any]:
        union_schema = core_schema
        while union_schema["type"] not in ("union", "nullable"):
            union_schema = union_schema["schema"]
        branches: List[Dict[str, Any]] = []
        # Pydantic folds a None branch into a nullable wrapper around the rest.
        if union_schema["type"] == "nullable":
            branches.append({"type": "null"})
            union_schema = union_schema["schema"]
        choices = union_schema["choices"] if union_schema["type"] == "union" else [union_schema]
        choices = [choice[0] if isinstance(choice, tuple) else choice for choice in choices]
        generate = handler.generate_json_schema.generate_inner
        return {"oneOf": branches + [generate(choice) for choice in choices]}


# JSON type mapping
JSON_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


_NUMERIC_CONSTRAINTS = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "multipleOf": "multiple_of",
}
_STRING_CONSTRAINTS = {"minLength": "min_length", "maxLength": "max_length"}
_ARRAY_CONSTRAINTS = {"minItems": "min_length", "maxItems": "max_length"}


def _constrained_value_type(python_type: Type, json_type: str, prop_schema: Dict[str, Any]) -> Type:
    """Attach supported constraints without changing the regex engine or accepting boolean bounds."""
    constraints = {}
    if json_type in {"integer", "number"}:
        for source, target in _NUMERIC_CONSTRAINTS.items():
            value = prop_schema.get(source)
            # Draft-04 boolean exclusive bounds are not numeric bounds.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                constraints[target] = value
    else:
        keywords = _STRING_CONSTRAINTS if json_type == "string" else _ARRAY_CONSTRAINTS if json_type == "array" else {}
        constraints.update({target: prop_schema[source] for source, target in keywords.items() if source in prop_schema})
    if json_type == "string" and isinstance(prop_schema.get("pattern"), str):
        pattern = prop_schema["pattern"]
        if _pattern_adapter(pattern) is None:
            logger.warning("Unsupported JSON Schema pattern %r; skipping local pattern validation", pattern)
        else:
            constraints["pattern"] = pattern
    if constraints:
        # Recursive callers retain the value type but discard returned Field metadata.
        return Annotated[python_type, Field(**constraints)]
    return python_type


class SchemaTransformer:
    """Class for transforming JSON schemas to Pydantic models."""

    @staticmethod
    def _matches_scalar_type(value: Any, json_types: List[str]) -> bool:
        """Match JSON scalar types without treating Python bools as numbers."""
        if isinstance(value, bool):
            return "boolean" in json_types
        if isinstance(value, (int, float)):
            return "number" in json_types or ("integer" in json_types and (isinstance(value, int) or value.is_integer()))
        return (value is None and "null" in json_types) or (isinstance(value, str) and "string" in json_types)

    @staticmethod
    def _scalar_choice_type(prop_schema: Dict[str, Any], choices: List[Any], root_schema: Dict[str, Any]) -> Type:
        """Keep only choices satisfying all advertised constraints, including compositions."""
        declared_type = prop_schema.get("type")
        json_types = declared_type if isinstance(declared_type, list) else [declared_type]
        # The finite set can be intersected once, without coercing values through union types.
        # Bind local references to the original root, and never retrieve remote schemas.
        validator = _ScalarChoiceValidator(root_schema, registry=Registry()).evolve(schema=prop_schema)
        choices = [value for value in choices if validator.is_valid(value)]
        if not choices:
            raise ValueError("Scalar choices do not match the declared JSON Schema constraints")

        def validate_choice(value: Any) -> Any:
            if declared_type is not None and not SchemaTransformer._matches_scalar_type(value, json_types):
                raise ValueError("Value does not match the declared JSON Schema type")
            for choice in choices:
                if isinstance(value, bool) == isinstance(choice, bool) and value == choice:
                    # Give Literal the matching choice so 1.0 cannot resolve to True.
                    return choice
            raise ValueError("Value does not match the JSON Schema choices")

        choice_type = Annotated[Literal[tuple(choices)], BeforeValidator(validate_choice)]
        if declared_type is not None:
            # Keep type metadata inside the choice branch when Optional wraps it.
            choice_type = Annotated[choice_type, Field(json_schema_extra={"type": declared_type})]
        return choice_type

    @staticmethod
    def _one_of_type(union_type: Any, branch_schemas: List[Any], root_schema: Dict[str, Any]) -> Any:
        """Require exactly one oneOf branch to match, which a Pydantic Union cannot express."""
        validator = _OneOfBranchValidator(root_schema, registry=Registry())
        branch_validators = [validator.evolve(schema=branch_schema) for branch_schema in branch_schemas]

        def validate_one_of(value: Any) -> Any:
            json_value = to_jsonable_python(value)
            matches = sum(branch_validator.is_valid(json_value) for branch_validator in branch_validators)
            if matches != 1:
                raise ValueError(f"Value matches {matches} oneOf branches, expected exactly one")
            return value

        return Annotated[union_type, BeforeValidator(validate_one_of), _OneOfJsonSchema()]

    @staticmethod
    def _resolve_ref(ref_path: str, root_schema: Dict[str, Any], model_cache: Dict[str, Type]) -> Type:
        """Resolve a $ref to its Pydantic-compatible value type."""
        if ref_path in model_cache:
            return model_cache[ref_path]

        # Resolve the complete local pointer, including escaped tokens and array
        # indices. An empty registry never retrieves remote schemas.
        resource = Resource.from_contents(root_schema, default_specification=DRAFT202012)
        try:
            ref_schema = Registry().resolver_with_root(resource).lookup(ref_path).contents
        except Unresolvable:
            logger.warning(f"Could not resolve $ref: {ref_path}")
            return Any

        # Avoid infinite recursion by adding a placeholder before conversion.
        model_cache[ref_path] = Any
        if isinstance(ref_schema, dict) and (
            ref_schema.get("properties")
            or (ref_schema.get("type") == "object" and ref_schema.get("additionalProperties") is False)
        ):
            ref_name = ref_path.split("/")[-1].replace("~1", "/").replace("~0", "~")
            model_name = ref_schema.get("title", ref_name)
            model = SchemaTransformer._create_nested_model(ref_schema, model_name, root_schema, model_cache)
        else:
            model, _ = SchemaTransformer.json_to_pydantic_field(ref_schema, True, root_schema, model_cache)
        model_cache[ref_path] = model
        return model

    @staticmethod
    def _create_nested_model(
        schema: Dict[str, Any], model_name: str, root_schema: Dict[str, Any], model_cache: Dict[str, Type]
    ) -> Type:
        """Create a nested Pydantic model from a schema."""
        fields = {}
        required_fields = set(schema.get("required", []))
        properties = schema.get("properties", {})

        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required_fields
            fields[prop_name] = SchemaTransformer.json_to_pydantic_field(prop_schema, is_required, root_schema, model_cache)

        return create_model(model_name, **fields)

    @staticmethod
    def json_to_pydantic_field(
        prop_schema: Union[Dict[str, Any], bool],
        required: bool,
        root_schema: Optional[Dict[str, Any]] = None,
        model_cache: Optional[Dict[str, Type]] = None,
    ) -> Tuple[Type, Field]:
        """
        Convert a JSON schema property to a Pydantic field.

        Args:
            prop_schema: JSON schema for the property
            required: Whether the field is required
            root_schema: Full root schema for resolving $refs
            model_cache: Cache for resolved models

        Returns:
            Tuple of (type, Field)
        """
        if isinstance(prop_schema, bool):
            value_type = Any if prop_schema else _FalseSchemaValue
            return value_type, Field(default=... if required else None)

        if root_schema is None:
            root_schema = {}
        if model_cache is None:
            model_cache = {}

        description = prop_schema.get("description")
        default = prop_schema.get("default")
        python_type: Any = Any

        # Handle $ref
        if "$ref" in prop_schema:
            python_type = SchemaTransformer._resolve_ref(prop_schema["$ref"], root_schema, model_cache)
        # Scalar value constraints must also appear in the model-facing schema.
        elif "const" in prop_schema and isinstance(prop_schema["const"], (str, int, float, bool, type(None))):
            python_type = SchemaTransformer._scalar_choice_type(prop_schema, [prop_schema["const"]], root_schema)
        elif prop_schema.get("enum") and all(
            isinstance(value, (str, int, float, bool, type(None))) for value in prop_schema["enum"]
        ):
            python_type = SchemaTransformer._scalar_choice_type(prop_schema, prop_schema["enum"], root_schema)
        # Handle oneOf/anyOf (unions)
        elif "oneOf" in prop_schema or "anyOf" in prop_schema:
            union_schemas = prop_schema.get("oneOf", prop_schema.get("anyOf", []))
            if union_schemas:
                union_types = []
                for union_schema in union_schemas:
                    member_type, _ = SchemaTransformer.json_to_pydantic_field(union_schema, True, root_schema, model_cache)
                    union_types.append(member_type)

                if "oneOf" in prop_schema:
                    # Keep repeated branches apart, which Union would merge, so they still count twice.
                    union_types = [
                        Annotated[member_type, object()] if member_type in union_types[:index] else member_type
                        for index, member_type in enumerate(union_types)
                    ]
                if len(union_types) == 1:
                    python_type = union_types[0]
                else:
                    python_type = Union[tuple(union_types)]
                    if "oneOf" in prop_schema:
                        python_type = SchemaTransformer._one_of_type(python_type, union_schemas, root_schema)
        # A JSON Schema type array is a union of its listed types.
        elif isinstance(prop_schema.get("type"), list):
            union_types = [
                SchemaTransformer.json_to_pydantic_field({**prop_schema, "type": json_type}, True, root_schema, model_cache)[0]
                for json_type in prop_schema["type"]
            ]
            python_type = Union[tuple(union_types)]
        # Handle regular types
        else:
            json_type = prop_schema.get("type")
            if json_type in JSON_TYPE_MAP:
                python_type = JSON_TYPE_MAP[json_type]

                if json_type == "array":
                    items_schema = prop_schema.get("items", {})
                    item_type, _ = SchemaTransformer.json_to_pydantic_field(items_schema, True, root_schema, model_cache)
                    python_type = List[item_type]

                elif json_type == "object":
                    python_type = Dict[str, Any]
                    value_schema = prop_schema.get("additionalProperties", {})
                    if isinstance(value_schema, dict):
                        value_type, _ = SchemaTransformer.json_to_pydantic_field(value_schema, True, root_schema, model_cache)
                        python_type = Dict[str, value_type]

                python_type = _constrained_value_type(python_type, json_type, prop_schema)

        field_kwargs = {"description": description}
        if required:
            field_kwargs["default"] = ...
        elif default is not None:
            field_kwargs["default"] = default
        else:
            if python_type is not _FalseSchemaValue:
                python_type = Optional[python_type]
            field_kwargs["default"] = None

        return (python_type, Field(**field_kwargs))

    @staticmethod
    def create_model_from_schema(
        schema: Dict[str, Any],
        model_name: str,
        tool_name_literal: str,
        docstring: Optional[str] = None,
        attribute_type: str = MCPAttributeType.TOOL,
        is_output_schema: bool = False,
    ) -> Type[BaseIOSchema]:
        """
        Dynamically create a Pydantic model from a JSON schema.

        Args:
            schema: JSON schema
            model_name: Name for the model
            tool_name_literal: Tool name to use for the Literal type
            docstring: Optional docstring for the model
            attribute_type: Type of MCP attribute (tool, resource, prompt)
            is_output_schema: If True, skip adding the tool_name/resource_name/prompt_name literal field.
                Output schemas represent tool responses and don't need an identifier field since
                the tool has already been selected and executed. Input schemas need the identifier
                for discriminated unions when selecting among multiple tools in an orchestrator.

        Returns:
            Pydantic model class
        """
        fields = {}
        required_fields = set(schema.get("required", []))
        properties = schema.get("properties")
        model_cache: Dict[str, Type] = {}

        if properties:
            for prop_name, prop_schema in properties.items():
                is_required = prop_name in required_fields
                fields[prop_name] = SchemaTransformer.json_to_pydantic_field(prop_schema, is_required, schema, model_cache)
        elif schema.get("type") == "object" and not properties:
            pass
        elif schema:
            logger.warning(
                f"Schema for {model_name} is not a typical object with properties. Fields might be empty beyond tool_name."
            )

        # Only add the attribute identifier field for input schemas
        if not is_output_schema:
            tool_name_type = cast(Type[str], Literal[tool_name_literal])
            fields[f"{attribute_type}_name"] = (
                tool_name_type,
                Field(..., description=f"Required identifier for the {tool_name_literal} {attribute_type}."),
            )

        # Create the model
        model = create_model(
            model_name,
            __base__=BaseIOSchema,
            __doc__=docstring or f"Dynamically generated Pydantic model for {model_name}",
            __config__={"title": tool_name_literal},
            **fields,
        )

        return model
