"""Offline Instructor clients using real OpenAI request and validation paths."""

from dataclasses import dataclass

import httpx
import instructor
import openai
from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails


@dataclass(frozen=True)
class ScriptedResponse:
    arguments: str = '{"answer": "42"}'
    input_tokens: int = 3
    output_tokens: int = 2
    cached_tokens: int = 1
    reasoning_tokens: int = 2
    schema_name: str = "Answer"

    def chat_usage(self) -> CompletionUsage:
        return CompletionUsage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=self.cached_tokens),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=self.reasoning_tokens),
        )

    def responses_body(self) -> dict[str, object]:
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "response-model",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": self.schema_name,
                    "arguments": self.arguments,
                    "status": "completed",
                }
            ],
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "input_tokens_details": {"cached_tokens": self.cached_tokens},
                "output_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
            },
        }

    def chat_body(self, mode: instructor.Mode) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant", "content": self.arguments}
        if mode == instructor.Mode.TOOLS:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": self.schema_name, "arguments": self.arguments},
                    }
                ],
            }
        return {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": "response-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": self.chat_usage().model_dump(),
        }


def response_transport(mode: instructor.Mode, responses: list[ScriptedResponse]) -> httpx.MockTransport:
    scripted = iter(responses)

    def respond(request: httpx.Request) -> httpx.Response:
        response = next(scripted)
        body = response.responses_body() if mode.value == "responses_tools" else response.chat_body(mode)
        expected_path = "/v1/responses" if mode.value == "responses_tools" else "/v1/chat/completions"
        if request.url.path != expected_path:
            raise AssertionError(f"Unexpected transport path: {request.url.path}")
        return httpx.Response(200, json=body)

    return httpx.MockTransport(respond)


def instructor_client(mode: instructor.Mode, responses: list[ScriptedResponse]) -> instructor.Instructor:
    return instructor.from_openai(
        openai.OpenAI(api_key="test", http_client=httpx.Client(transport=response_transport(mode, responses))), mode=mode
    )


def async_instructor_client(mode: instructor.Mode, responses: list[ScriptedResponse]) -> instructor.AsyncInstructor:
    return instructor.from_openai(
        openai.AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=response_transport(mode, responses))),
        mode=mode,
    )
