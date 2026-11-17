"""Keep nullable and union fields usable by DSPy's native typed adapter."""

import json
from pathlib import Path
import sys
from typing import List, Literal, Optional, Union

import dspy
from pydantic import TypeAdapter, create_model
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_agents import BaseIOSchema  # noqa: E402
from dspy_integration.bridge import DSPyAtomicModule  # noqa: E402


class Query(BaseIOSchema):
    """Query for a typed result."""

    text: str


class FixtureLM(dspy.LM):
    """Return deterministic completion text through the real DSPy adapter."""

    def __init__(self, value):
        super().__init__("openai/test", cache=False)
        self.value = value

    def __call__(self, *args, **kwargs):
        text = self.value if isinstance(self.value, str) else json.dumps(self.value)
        return ["[[ ## value ## ]]\n" + text + "\n\n[[ ## completed ## ]]"]


@pytest.mark.parametrize(
    "annotation,value",
    [
        (Optional[int], None),
        (int | None, None),
        (Optional[int], 7),
        (int | None, 7),
        (Union[int, str], 42),
        (Union[int, str], "answer"),
        (Optional[bool], False),
        (Optional[List[int]], [1, 2]),
        (Optional[List[int]], None),
        (List[Optional[int]], [1, None, 3]),
        (Optional[Literal["yes", "no"]], None),
        (str | None, "answer"),
        (int | str, 42),
        (List[int], [1, 2]),
    ],
)
def test_native_predictions_preserve_declared_union_values(annotation, value):
    output = create_model("Result", __base__=BaseIOSchema, __doc__="Typed result.", value=(annotation, ...))
    module = DSPyAtomicModule(Query, output, use_chain_of_thought=False)

    with dspy.context(lm=FixtureLM(value)):
        prediction = module(text="Produce the configured value")

    assert type(prediction.value) is type(value)
    assert prediction.value == value
    actual_type = module.signature.model_fields["value"].annotation
    assert TypeAdapter(actual_type).json_schema() == TypeAdapter(annotation).json_schema()


@pytest.mark.parametrize("annotation", [Optional[int], int | None, Optional[List[int]]])
def test_run_validated_accepts_null_for_optional_outputs(annotation):
    output = create_model("NullableResult", __base__=BaseIOSchema, __doc__="Nullable result.", value=(annotation, ...))
    module = DSPyAtomicModule(Query, output, use_chain_of_thought=False)

    with dspy.context(lm=FixtureLM(None)):
        result = module.run_validated(text="No value is available")

    assert isinstance(result, output)
    assert result.value is None
