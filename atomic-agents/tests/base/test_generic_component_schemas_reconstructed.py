"""New regression coverage for the preserved generic-component schema repair.

Written during recovery on 2026-11-05; this is not the lost original test file.
"""

from typing import Generic, TypeVar

import pytest

from atomic_agents.base.base_io_schema import BaseIOSchema
from atomic_agents.base.base_prompt import BasePrompt, BasePromptConfig
from atomic_agents.base.base_resource import BaseResource, BaseResourceConfig
from atomic_agents.base.base_tool import BaseTool, BaseToolConfig


class Request(BaseIOSchema):
    """Text to process."""

    text: str


class Response(BaseIOSchema):
    """Processed text."""

    text: str


class OtherRequest(BaseIOSchema):
    """Alternative request text."""

    text: str


class OtherResponse(BaseIOSchema):
    """Alternative response text."""

    text: str


class Operations:
    """Concrete local operations shared by the three component interfaces."""

    def run(self, params):
        return self.output_schema(text=params.text.upper())

    def read(self, params):
        return self.output_schema(text=params.text.upper())

    def generate(self, params):
        return self.output_schema(text=params.text.upper())


@pytest.fixture(
    params=[
        (BaseTool, BaseToolConfig, "tool", "run"),
        (BaseResource, BaseResourceConfig, "resource", "read"),
        (BasePrompt, BasePromptConfig, "prompt", "generate"),
    ],
    ids=["tool", "resource", "prompt"],
)
def component(request):
    return request.param


def assert_contract(instance, component, input_schema=Request, output_schema=Response):
    """Check public schema, metadata, and operation behavior together."""
    _, _, name_prefix, operation = component
    assert instance.input_schema is input_schema
    assert instance.output_schema is output_schema
    assert getattr(instance, f"{name_prefix}_name") == input_schema.model_json_schema()["title"]
    assert getattr(instance, f"{name_prefix}_description") == input_schema.model_json_schema()["description"]
    result = getattr(instance, operation)(input_schema(text="hello"))
    assert type(result) is output_schema
    assert result.text == "HELLO"


def test_direct_concrete_component(component):
    base, *_ = component

    class Concrete(Operations, base[Request, Response]):
        pass

    assert_contract(Concrete(), component)


def test_concrete_component_plain_descendant(component):
    base, *_ = component

    class Concrete(Operations, base[Request, Response]):
        pass

    class Child(Concrete):
        pass

    assert_contract(Child(), component)


def test_dynamic_two_parameter_specialization(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    assert_contract(Reusable[Request, Response](), component)


def test_concrete_generic_specialization(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class Concrete(Reusable[Request, Response]):
        pass

    assert_contract(Concrete(), component)


def test_plain_descendant_of_generic_specialization(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class Concrete(Reusable[Request, Response]):
        pass

    class Child(Concrete):
        pass

    class Grandchild(Child):
        pass

    assert_contract(Grandchild(), component)


def test_swapped_parameter_order(component):
    base, *_ = component

    class Reversed[Out: BaseIOSchema, In: BaseIOSchema](Operations, base[In, Out]):
        pass

    assert_contract(Reversed[Response, Request](), component)


def test_fixed_output_parameter(component):
    base, *_ = component

    class FixedOutput[In: BaseIOSchema](Operations, base[In, Response]):
        pass

    assert_contract(FixedOutput[Request](), component)


def test_fixed_input_parameter(component):
    base, *_ = component

    class FixedInput[Out: BaseIOSchema](Operations, base[Request, Out]):
        pass

    assert_contract(FixedInput[Response](), component)


def test_two_forwarding_generic_layers(component):
    base, *_ = component

    class First[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class Second[In: BaseIOSchema, Out: BaseIOSchema](First[In, Out]):
        pass

    assert_contract(Second[Request, Response](), component)


def test_partial_specialization_through_intermediate_layer(component):
    base, *_ = component

    class First[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class Second[In: BaseIOSchema](First[In, Response]):
        pass

    class Concrete(Second[Request]):
        pass

    assert_contract(Concrete(), component)


def test_reordered_intermediate_parameters(component):
    base, *_ = component

    class First[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class Second[Out: BaseIOSchema, In: BaseIOSchema](First[In, Out]):
        pass

    class Concrete(Second[Response, Request]):
        pass

    assert_contract(Concrete(), component)


def test_sibling_specializations_do_not_overwrite_each_other(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    first = Reusable[Request, Response]()
    second = Reusable[OtherRequest, OtherResponse]()
    assert_contract(first, component)
    assert_contract(second, component, OtherRequest, OtherResponse)
    assert_contract(first, component)


def test_concrete_sibling_specializations_are_isolated(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    class First(Reusable[Request, Response]):
        pass

    class Second(Reusable[OtherRequest, OtherResponse]):
        pass

    assert_contract(Second(), component, OtherRequest, OtherResponse)
    assert_contract(First(), component)


def test_legacy_generic_without_forwarded_base_arguments(component):
    base, *_ = component

    class Legacy[In: BaseIOSchema, Out: BaseIOSchema](Operations, base):
        pass

    assert_contract(Legacy[Request, Response](), component)


def test_untyped_component_defaults(component):
    base, _, prefix, _ = component

    class Untyped(Operations, base):
        pass

    instance = Untyped()
    assert instance.input_schema is BaseIOSchema
    assert instance.output_schema is BaseIOSchema
    assert getattr(instance, f"{prefix}_name") == "BaseIOSchema"


def test_legacy_unspecialized_component_defaults(component):
    base, *_ = component

    class Legacy[In: BaseIOSchema, Out: BaseIOSchema](Operations, base):
        pass

    assert Legacy().input_schema is BaseIOSchema
    assert Legacy().output_schema is BaseIOSchema


def test_explicit_base_redeclaration_takes_precedence(component):
    base, *_ = component

    class Original(Operations, base[OtherRequest, OtherResponse]):
        pass

    class Override(Original, base[Request, Response]):
        pass

    assert_contract(Override(), component)
    assert_contract(Original(), component, OtherRequest, OtherResponse)


def test_descendant_retains_explicit_base_override(component):
    base, *_ = component

    class Original(Operations, base[OtherRequest, OtherResponse]):
        pass

    class Override(Original, base[Request, Response]):
        pass

    class Child(Override):
        pass

    assert_contract(Child(), component)


def test_generic_mixin_does_not_replace_component_arguments(component):
    base, *_ = component
    Extra = TypeVar("Extra")

    class Mixin(Generic[Extra]):
        pass

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Mixin[str], Operations, base[In, Out]):
        pass

    assert_contract(Reusable[Request, Response](), component)


def test_custom_constructor_is_preserved(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        def __init__(self, marker):
            super().__init__()
            self.marker = marker

    instance = Reusable[Request, Response](marker="retained")
    assert instance.marker == "retained"
    assert_contract(instance, component)


def test_concrete_specialization_schemas_available_during_constructor(component):
    base, *_ = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        def __init__(self):
            super().__init__()
            self.schemas_at_init = (self.input_schema, self.output_schema)

    class Concrete(Reusable[Request, Response]):
        pass

    instance = Concrete()
    assert instance.schemas_at_init == (Request, Response)
    assert_contract(instance, component)


def test_config_overrides_metadata_without_changing_schemas(component):
    base, config_type, prefix, operation = component

    class Reusable[In: BaseIOSchema, Out: BaseIOSchema](Operations, base[In, Out]):
        pass

    instance = Reusable[Request, Response](config=config_type(title="Custom title", description="Custom description"))
    assert getattr(instance, f"{prefix}_name") == "Custom title"
    assert getattr(instance, f"{prefix}_description") == "Custom description"
    assert instance.input_schema is Request
    assert instance.output_schema is Response
    assert type(getattr(instance, operation)(Request(text="hello"))) is Response


def test_classic_generic_typevar_declaration(component):
    base, *_ = component
    InputType = TypeVar("InputType", bound=BaseIOSchema)
    OutputType = TypeVar("OutputType", bound=BaseIOSchema)

    class Reusable(Operations, base[InputType, OutputType], Generic[InputType, OutputType]):
        pass

    assert_contract(Reusable[Request, Response](), component)
