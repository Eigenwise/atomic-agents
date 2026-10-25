import os
import uuid

import instructor
import openai
from dotenv import load_dotenv
from openinference.instrumentation import using_session
from phoenix.otel import register
from pydantic import Field
from rich.console import Console

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import SystemPromptGenerator
from phoenix_tracing.tools.calculator import CalculatorTool, CalculatorToolInputSchema

load_dotenv()

os.environ.setdefault("ATOMIC_AGENTS_TRACE_LLM_SPANS", "false")
tracer_provider = register(project_name="atomic-agents-example", auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)
console = Console()


class CalculationPlanSchema(BaseIOSchema):
    """A single arithmetic expression that answers the user's question."""

    expression: str = Field(..., description="A sympy-compatible arithmetic expression, for example '12 * 7 + 3'.")
    reasoning: str = Field(..., description="One sentence on why this expression answers the question.")


class CalculationResultSchema(BaseIOSchema):
    """The user's question together with the evaluated expression."""

    question: str = Field(..., description="The original question.")
    expression: str = Field(..., description="The expression that was evaluated.")
    result: str = Field(..., description="The evaluated result.")


def build_client() -> instructor.Instructor:
    return instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def build_planner(client: instructor.Instructor) -> AtomicAgent[BasicChatInputSchema, CalculationPlanSchema]:
    return AtomicAgent[BasicChatInputSchema, CalculationPlanSchema](
        AgentConfig(
            client=client,
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            system_prompt_generator=SystemPromptGenerator(
                background=["You turn word problems into one arithmetic expression a calculator can evaluate."],
                output_instructions=["Use only numbers and the operators + - * / ** ( )."],
            ),
        )
    )


def build_answerer(client: instructor.Instructor) -> AtomicAgent[CalculationResultSchema, BasicChatOutputSchema]:
    return AtomicAgent[CalculationResultSchema, BasicChatOutputSchema](
        AgentConfig(
            client=client,
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            system_prompt_generator=SystemPromptGenerator(
                background=["You explain a computed result to the user in one friendly sentence."],
            ),
        )
    )


def answer_question(question: str) -> str:
    client = build_client()
    planner = build_planner(client)
    calculator = CalculatorTool()
    answerer = build_answerer(client)

    with tracer.start_as_current_span("answer_question", openinference_span_kind="chain") as turn:
        turn.set_input(question)
        plan = planner.run(BasicChatInputSchema(chat_message=question))
        calculation = calculator.run(CalculatorToolInputSchema(expression=plan.expression))
        answer = answerer.run(
            CalculationResultSchema(question=question, expression=plan.expression, result=calculation.result)
        )
        turn.set_output(answer.chat_message)
        return answer.chat_message


def main() -> None:
    question = "If 3 friends split a 47 euro bill and add a 15 percent tip, how much does each pay?"
    with using_session(session_id=str(uuid.uuid4())):
        answer = answer_question(question)

    console.print(f"[bold green]Question:[/bold green] {question}")
    console.print(f"[bold blue]Answer:[/bold blue] {answer}")
    console.print(
        "\nOpen Phoenix at "
        f"[link]{os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')}[/link] "
        "and look at the 'atomic-agents-example' project."
    )


if __name__ == "__main__":
    main()
