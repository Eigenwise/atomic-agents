"""
Stage 2: Raw Atomic Agents with Manual Prompts.

This module demonstrates Atomic Agents' capabilities:
- Full Pydantic ecosystem with runtime validation
- Instructor integration for robust structured outputs
- Guaranteed schema compliance

Limitations shown:
- Manual prompt engineering (guesswork)
- No systematic way to improve prompts
- No automatic few-shot selection

Design: Single function entry point, internal helpers follow SRP.
"""

import time
from typing import Any, Dict, List, Tuple

import instructor
import openai

from atomic_agents.agents.atomic_agent import AgentConfig, AtomicAgent
from atomic_agents.context.system_prompt_generator import SystemPromptGenerator

from dspy_integration.domain.models import (
    GENRES,
    MovieGenreOutput,
    MovieReviewInput,
    EvalResult,
)
from dspy_integration.domain.datasets import TEST_DATASET
from dspy_integration.domain.evaluation import evaluate_predictions
from dspy_integration.presentation.console import (
    console,
    display_stage_header,
    display_panel,
    display_code,
    display_step_header,
    display_success,
    display_results_table,
    create_progress_context,
)


# =============================================================================
# CODE EXAMPLES FOR DISPLAY
# =============================================================================


SCHEMA_CODE_EXAMPLE = '''class MovieGenreOutput(BaseIOSchema):
    """Output schema for movie genre classification."""

    genre: Literal["action", "comedy", "drama", "horror", "sci-fi", "romance"] = Field(
        ...,
        description="The primary genre of the movie.",
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,  # VALIDATED! Must be between 0 and 1
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation for the classification.",
    )

# The LLM output MUST match this schema or it fails validation.
# No more parsing "high" vs "0.85" vs "85%" - it's always a float!'''


# =============================================================================
# MAIN STAGE FUNCTION
# =============================================================================


def run_stage2_raw_atomic_agents(api_key: str) -> Tuple[EvalResult, Dict[str, Any]]:
    """
    Run Stage 2: Raw Atomic Agents demonstration.

    This demonstrates Atomic Agents' beautiful structured outputs,
    but with manual prompt engineering.

    Args:
        api_key: OpenAI API key

    Returns:
        Tuple of (evaluation results, behind-the-scenes data)
    """
    display_stage_header("STAGE 2: Raw Atomic Agents", "magenta")
    _display_stage_overview()

    # Step 1: Show Pydantic schema
    _display_schema_explanation()

    # Step 2: Show manual system prompt
    system_prompt = _create_system_prompt()
    generated_prompt = system_prompt.generate_prompt()
    _display_manual_prompt(generated_prompt)
    _display_manual_prompt_problem()

    # Step 3: Create agent
    agent = _create_agent(api_key, system_prompt)

    # Step 4: Show schema enforcement
    _display_schema_enforcement()

    # Step 5: Evaluate
    eval_result, predictions = _evaluate_agent(agent)

    # Step 6: Display results
    _display_stage_results(eval_result, predictions)

    behind_scenes = {
        "system_prompt": generated_prompt,
        "schema_enforced": True,
        "manual_engineering": True,
    }

    return eval_result, behind_scenes


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def _display_stage_overview() -> None:
    """Display stage 2 overview panel."""
    content = """[green]ATOMIC AGENTS STRENGTHS:[/green]
• Full Pydantic ecosystem (validators, serializers, Field constraints)
• Instructor integration for robust structured output
• Python-native type safety with runtime validation
• ge/le constraints on confidence (guaranteed 0-1)

[yellow]LIMITATIONS:[/yellow]
• Manual prompt engineering - no automatic optimization
• No systematic few-shot example selection
• Prompt improvements require guesswork and iteration"""

    display_panel(content, "Stage 2 Overview", "magenta")


def _display_schema_explanation() -> None:
    """Display explanation of Pydantic schemas."""
    display_step_header("Step 2.1: Define Pydantic Schema")
    console.print("Atomic Agents uses Pydantic for type-safe outputs:\n")
    display_code(SCHEMA_CODE_EXAMPLE)


def _display_manual_prompt(generated_prompt: str) -> None:
    """Display the manually crafted system prompt."""
    display_step_header("Step 2.2: Manual System Prompt (The Guesswork)")

    content = "[dim]This is the system prompt WE WROTE BY HAND:[/dim]\n\n" + generated_prompt
    display_panel(content, "Manual System Prompt (Our Best Guess)", "yellow")


def _display_manual_prompt_problem() -> None:
    """Display the problem with manual prompt engineering."""
    content = """[red]THE PROBLEM:[/red]

We wrote this prompt based on intuition. Questions we can't answer:

• Is 'Be decisive' helping or hurting accuracy?
• Should we add few-shot examples? Which ones?
• Is the step-by-step instruction actually useful?
• Would different wording improve results?

[yellow]Without DSPy, we're just guessing![/yellow]"""

    display_panel(content, "The Manual Prompt Engineering Problem", "red")


def _display_schema_enforcement() -> None:
    """Display how schema enforcement works."""
    display_step_header("Step 2.4: Schema Enforcement in Action")

    content = """[cyan]What happens under the hood:[/cyan]

1. Atomic Agents sends your prompt + Pydantic schema to the LLM
2. Instructor (the library) converts schema to JSON Schema for the LLM
3. LLM generates output attempting to match the schema
4. Instructor validates the response against Pydantic
5. If validation fails, Instructor retries with error feedback
6. You get a guaranteed-valid Pydantic object or an exception

[green]Result:[/green] genre is ALWAYS one of our 6 options,
confidence is ALWAYS a float between 0 and 1!"""

    display_panel(content, "How Schema Enforcement Works", "cyan")


# =============================================================================
# AGENT CREATION
# =============================================================================


def _create_system_prompt() -> SystemPromptGenerator:
    """Create the manually crafted system prompt."""
    return SystemPromptGenerator(
        background=[
            "You are a movie genre classification expert.",
            "You analyze movie reviews and determine the primary genre.",
            f"Valid genres are: {', '.join(GENRES)}",
        ],
        steps=[
            "Read the review carefully.",
            "Identify key genre indicators (action words, emotional language, etc.).",
            "Consider the overall tone and subject matter.",
            "Select the single most appropriate genre.",
            "Provide a confidence score based on how clear the genre signals are.",
        ],
        output_instructions=[
            "Be decisive - pick ONE primary genre even if multiple could apply.",
            "Confidence should be 0.7-1.0 for clear cases, 0.5-0.7 for ambiguous ones.",
            "Keep reasoning brief but specific to the review.",
        ],
    )


def _create_agent(
    api_key: str,
    system_prompt: SystemPromptGenerator,
) -> AtomicAgent:
    """Create the Atomic Agent with schema validation."""
    display_step_header("Step 2.3: Create Atomic Agent")

    client = instructor.from_openai(openai.OpenAI(api_key=api_key))

    agent = AtomicAgent[MovieReviewInput, MovieGenreOutput](
        config=AgentConfig(
            client=client,
            model="gpt-5.6-terra",
            system_prompt_generator=system_prompt,
        )
    )

    display_success("Agent created with schema validation")
    return agent


# =============================================================================
# EVALUATION HELPERS
# =============================================================================


def _evaluate_agent(
    agent: AtomicAgent,
) -> Tuple[EvalResult, List[Dict[str, Any]]]:
    """Evaluate the agent on test set."""
    display_step_header("Step 2.5: Evaluation on Test Set")

    predictions = []
    start_time = time.time()

    with create_progress_context("[magenta]Running predictions...") as progress:
        task = progress.add_task("Predicting...", total=len(TEST_DATASET))

        for test_ex in TEST_DATASET:
            prediction = _get_single_prediction(agent, test_ex)
            predictions.append(prediction)
            progress.advance(task)

    elapsed = time.time() - start_time
    eval_result = evaluate_predictions(predictions, TEST_DATASET)
    eval_result.avg_time = elapsed / len(TEST_DATASET)

    return eval_result, predictions


def _get_single_prediction(
    agent: AtomicAgent,
    test_example: Dict[str, str],
) -> Dict[str, Any]:
    """Get a single prediction from the agent."""
    try:
        result = agent.run(MovieReviewInput(review=test_example["review"]))
        return {
            "genre": result.genre,  # Already validated by Pydantic!
            "confidence": result.confidence,  # Already a float!
            "reasoning": result.reasoning,
        }
    except Exception as e:
        return {
            "genre": "error",
            "confidence": 0,
            "reasoning": str(e),
        }


# =============================================================================
# RESULTS DISPLAY
# =============================================================================


def _display_stage_results(
    eval_result: EvalResult,
    predictions: List[Dict[str, Any]],
) -> None:
    """Display stage 2 results and analysis."""
    display_step_header("Step 2.6: The Benefit - Type-Safe Outputs")

    # Show sample outputs
    samples = "\n".join(
        [
            f"  • genre='{predictions[i]['genre']}' (Literal) " f"confidence={predictions[i]['confidence']:.2f} (float)"
            for i in range(min(3, len(predictions)))
        ]
    )

    content = f"""[green]ATOMIC AGENTS ADVANTAGE:[/green]

Look at these outputs - perfectly structured:
{samples}

[cyan]Benefits:[/cyan]
• genre is guaranteed to be one of our 6 valid options
• confidence is always a float between 0.0 and 1.0
• No parsing needed - direct attribute access
• IDE autocomplete works perfectly
• Downstream code can trust the types"""

    display_panel(content, "Structured Output Benefits", "green")
    display_results_table(eval_result, "Stage 2 Results", show_confidence=True)
