"""Input and target fields must not overwrite each other in DSPy's flat namespace."""

import sys
from pathlib import Path

import pytest
from pydantic import Field, create_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_integration.bridge import (  # noqa: E402
    DSPyAtomicModule,
    create_dspy_example,
    create_dspy_signature_from_schemas,
)

from atomic_agents.base.base_io_schema import BaseIOSchema  # noqa: E402


@pytest.mark.parametrize("names", [("text",), ("text", "score")])
@pytest.mark.parametrize("entrypoint", ["signature", "module", "example"])
def test_overlapping_fields_raise_before_input_or_target_is_lost(names, entrypoint):
    fields = {name: (str, ...) for name in names}
    input_schema = create_model("Original", __base__=BaseIOSchema, __doc__="Original source fields.", **fields)
    output_schema = create_model("Target", __base__=BaseIOSchema, __doc__="Expected target fields.", **fields)

    with pytest.raises(ValueError, match="distinct field names") as error:
        if entrypoint == "signature":
            create_dspy_signature_from_schemas(input_schema, output_schema)
        elif entrypoint == "module":
            DSPyAtomicModule(input_schema, output_schema)
        else:
            create_dspy_example(
                input_schema,
                output_schema,
                {name: "original input" for name in names},
                {name: "expected answer" for name in names},
            )

    assert str(error.value).endswith(", ".join(sorted(names)))


def test_defaulted_fields_are_still_part_of_the_shared_namespace():
    class Original(BaseIOSchema):
        """Original source text."""

        text: str = "original input"

    class Target(BaseIOSchema):
        """Expected replacement text."""

        text: str = "expected answer"

    with pytest.raises(ValueError, match="distinct field names"):
        create_dspy_example(Original, Target, {}, {})


def test_distinct_fields_preserve_native_signature_and_training_roles():
    class Original(BaseIOSchema):
        """Original source text."""

        text: str

    class Target(BaseIOSchema):
        """Expected replacement text."""

        rewritten_text: str

    signature = create_dspy_signature_from_schemas(Original, Target)
    example = create_dspy_example(Original, Target, {"text": "original input"}, {"rewritten_text": "expected answer"})

    assert list(signature.input_fields) == ["text"]
    assert list(signature.output_fields) == ["rewritten_text"]
    assert dict(example.inputs()) == {"text": "original input"}
    assert dict(example.labels()) == {"rewritten_text": "expected answer"}


def test_matching_aliases_do_not_collide_when_python_field_names_differ():
    class Original(BaseIOSchema):
        """Input with a wire alias."""

        source: str = Field(alias="text")

    class Target(BaseIOSchema):
        """Output with the same wire alias."""

        answer: str = Field(alias="text")

    signature = create_dspy_signature_from_schemas(Original, Target)
    example = create_dspy_example(Original, Target, {"text": "original input"}, {"text": "expected answer"})

    assert list(signature.input_fields) == ["source"]
    assert list(signature.output_fields) == ["answer"]
    assert dict(example.inputs()) == {"source": "original input"}
    assert dict(example.labels()) == {"answer": "expected answer"}
