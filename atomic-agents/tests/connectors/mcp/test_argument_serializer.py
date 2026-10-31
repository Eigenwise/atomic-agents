from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from atomic_agents.connectors.mcp.argument_serializer import ArgumentSerializer


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("kind, expected", [("nullable", {"kind": "nullable", "name": None}), ("plain", {"kind": "plain"})])
def test_structured_union_preserves_null_only_in_the_matching_branch(keyword, kind, expected):
    schema = {
        keyword: [
            {
                "type": "object",
                "properties": {"kind": {"const": "nullable"}, "name": {"type": ["string", "null"]}},
                "required": ["kind"],
            },
            {
                "type": "object",
                "properties": {"kind": {"const": "plain"}, "name": {"type": "string"}},
                "required": ["kind"],
            },
        ]
    }
    arguments = {"kind": kind, "name": None}
    original = deepcopy(arguments)

    result = ArgumentSerializer(schema).serialize(arguments)

    assert result == expected
    Draft202012Validator(schema).validate(result)
    assert arguments == original


def test_recursive_object_reference_filters_each_level_without_mutating_input():
    schema = {
        "type": "object",
        "properties": {"tree": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
    }
    arguments = {"tree": {"name": None, "children": [{"name": None}, {"name": "leaf"}]}}
    original = deepcopy(arguments)

    result = ArgumentSerializer(schema).serialize(arguments)

    assert result == {"tree": {"children": [{}, {"name": "leaf"}]}}
    Draft202012Validator(schema).validate(result)
    assert arguments == original


@pytest.mark.parametrize("reference", ["#/$defs/Missing", "https://example.invalid/schema.json"])
def test_unresolved_reference_keeps_transformer_any_fallback_without_remote_resolution(reference):
    schema = {"type": "object", "properties": {"value": {"$ref": reference}}}
    assert ArgumentSerializer(schema).serialize({"value": None}) == {"value": None}


@pytest.mark.parametrize("nullable", [True, False])
def test_allof_applies_each_property_null_restriction(nullable):
    schema = {
        "allOf": [
            {"properties": {"name": {"type": ["string", "null"]}}},
            {"properties": {"name": {"type": ["string", "null"] if nullable else "string"}}},
        ]
    }
    result = ArgumentSerializer(schema).serialize({"name": None})
    assert result == ({"name": None} if nullable else {})
    Draft202012Validator(schema).validate(result)


def test_invalid_structured_union_does_not_pick_an_arbitrary_branch():
    schema = {
        "anyOf": [
            {"properties": {"kind": {"const": "first"}, "name": {"type": "string"}}},
            {"properties": {"kind": {"const": "second"}, "name": {"type": ["string", "null"]}}},
        ]
    }
    arguments = {"kind": "unknown", "name": None}
    assert ArgumentSerializer(schema).serialize(arguments) == arguments


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("reverse", [False, True])
def test_valid_union_input_keeps_explicit_null_independently_of_branch_order(keyword, reverse):
    branches = [
        {"type": "object", "properties": {"name": {"type": "string"}}},
        {"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
    ]
    schema = {keyword: branches[::-1] if reverse else branches}
    arguments = {"name": None}

    result = ArgumentSerializer(schema).serialize(arguments)

    assert result == arguments
    Draft202012Validator(schema).validate(result)


@pytest.mark.parametrize("reference", ["#/$defs/Missing", "https://example.invalid/schema.json"])
@pytest.mark.parametrize("ref_first", [True, False])
@pytest.mark.parametrize("constraint", [{"type": "string"}, {"enum": ["brief"]}, {"const": "brief"}])
def test_unresolved_reference_still_enforces_sibling_constraints(reference, ref_first, constraint):
    value_schema = {"$ref": reference, **constraint} if ref_first else {**constraint, "$ref": reference}
    schema = {"type": "object", "properties": {"value": value_schema}}
    assert ArgumentSerializer(schema).serialize({"value": None}) == {}
    assert ArgumentSerializer(schema).serialize({"value": "brief"}) == {"value": "brief"}


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_unresolved_reference_in_composition_keeps_known_constraints(keyword):
    schema = {
        "type": "object",
        "properties": {"value": {keyword: [{"$ref": "https://example.invalid/schema.json", "type": "string"}]}},
    }
    assert ArgumentSerializer(schema).serialize({"value": None}) == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_valid_branch_is_preferred_even_when_common_constraint_needs_filtering(reverse):
    branches = [
        {"properties": {"name": {"type": "string"}}},
        {"properties": {"name": {"type": ["string", "null"]}}},
    ]
    schema = {
        "properties": {"other": {"type": "string"}},
        "anyOf": branches[::-1] if reverse else branches,
    }
    result = ArgumentSerializer(schema).serialize({"name": None, "other": None})
    assert result == {"name": None}
    Draft202012Validator(schema).validate(result)


def test_unresolved_reference_fallback_with_declared_root_dialect():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"value": {"$ref": "https://example.invalid/schema.json", "type": "string"}},
    }
    assert ArgumentSerializer(schema).serialize({"value": None}) == {}
