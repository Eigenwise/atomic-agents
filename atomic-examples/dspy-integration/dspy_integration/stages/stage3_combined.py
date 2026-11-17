"""
Stage 3: DSPy + Atomic Agents Combined.

This module demonstrates the best of both worlds:
- DSPy's automatic prompt optimization
- Atomic Agents' type-safe structured outputs

The bridge module connects both frameworks, enabling:
- Pydantic schemas as DSPy signatures
- DSPy optimizers for Atomic Agents
- Validated, optimized outputs

Design: Single function entry point, internal helpers follow SRP.
"""

import json
import time
from typing import Any, Dict, List, Tuple

import dspy

from dspy_integration.bridge import DSPyAtomicModule, create_dspy_example
from dspy_integration.domain.models import (
    MovieGenreOutput,
    MovieReviewInput,
    EvalResult,
)
from dspy_integration.domain.datasets import TRAINING_DATASET, TEST_DATASET
from dspy_integration.domain.evaluation import evaluate_predictions
from dspy_integration.presentation.console import (
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


BRIDGE_CODE_EXAMPLE = """# The bridge combines both frameworks:
module = DSPyAtomicModule(
    input_schema=MovieReviewInput,    # Pydantic input validation
    output_schema=MovieGenreOutput,   # Pydantic output structure
    instructions="Classify the movie review into a genre.",
    use_chain_of_thought=True,        # DSPy's reasoning capability
)

# Behind the scenes:
# 1. Pydantic schemas are converted to DSPy signatures
# 2. DSPy handles prompt construction and optimization
# 3. Outputs are validated against Pydantic schemas
# 4. You get type-safe results that DSPy optimized!"""


# =============================================================================
# MAIN STAGE FUNCTION
# =============================================================================


def run_stage3_combined(api_key: str) -> Tuple[EvalResult, Dict[str, Any]]:
    """
    Run Stage 3: Combined DSPy + Atomic Agents demonstration.

    This demonstrates the best of both worlds - DSPy optimization
    with Atomic Agents type safety.

    Args:
        api_key: OpenAI API key

    Returns:
        Tuple of (evaluation results, behind-the-scenes data)
    """
    display_stage_header("STAGE 3: DSPy + Atomic Agents", "green")
    _display_stage_overview()

    # Configure DSPy
    lm = dspy.LM("openai/gpt-5.6-terra", api_key=api_key)
    dspy.configure(lm=lm)

    # Step 1: Show bridge module
    _display_bridge_explanation()

    # Step 2: Create module
    module = _create_bridge_module()

    # Step 3: Show schema conversion
    _display_schema_conversion()

    # Step 4: Create training examples
    trainset = _create_training_set()

    # Step 5: Run optimization
    optimized_module = _run_optimization(module, trainset)

    # Step 6: Show optimized prompt
    optimized_prompt = _capture_optimized_prompt(lm, optimized_module)

    # Step 7: Evaluate
    eval_result, predictions = _evaluate_module(optimized_module)

    # Step 8: Display results
    _display_stage_results(eval_result)

    behind_scenes = {
        "optimized_prompt_sample": optimized_prompt[:1000] if optimized_prompt else "N/A",
        "schema_enforced": True,
        "dspy_optimized": True,
    }

    return eval_result, behind_scenes


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def _display_stage_overview() -> None:
    """Display stage 3 overview panel."""
    content = """[green]THE SOLUTION:[/green]
Combine DSPy's automatic optimization with Atomic Agents' type safety!

[cyan]WHAT WE GET:[/cyan]
• DSPy automatically finds the best prompts and examples
• Atomic Agents guarantees output structure
• Measurable improvements through optimization
• Production-ready typed outputs

[yellow]THE BEST OF BOTH WORLDS[/yellow]"""

    display_panel(content, "Stage 3 Overview", "green")


def _display_bridge_explanation() -> None:
    """Display explanation of the bridge module."""
    display_step_header("Step 3.1: The Bridge - DSPyAtomicModule")
    display_code(BRIDGE_CODE_EXAMPLE)


def _display_schema_conversion() -> None:
    """Display how schemas are converted to signatures."""
    display_step_header("Step 3.2: Schema-to-Signature Conversion")

    content = """[cyan]Pydantic Schema → DSPy Signature:[/cyan]

Input fields: review (str)
Output fields: genre (Literal), confidence (float), reasoning (str)

[dim]The bridge automatically converts Pydantic field descriptions
into DSPy field descriptors, preserving all metadata.[/dim]"""

    display_panel(content, "Automatic Conversion", "cyan")


def _display_training_explanation() -> None:
    """Display explanation of type-safe training examples."""
    display_step_header("Step 3.3: Type-Safe Training Examples")

    content = """[cyan]Creating training examples with validation:[/cyan]

Each example is validated against our Pydantic schemas!
If you accidentally put confidence=1.5 or genre='thriller',
you get an immediate error - not a silent failure later."""

    display_panel(content, "Validated Training Data", "cyan")


# =============================================================================
# MODULE CREATION
# =============================================================================


def _create_bridge_module() -> DSPyAtomicModule:
    """Create the DSPy-Atomic bridge module."""
    return DSPyAtomicModule(
        input_schema=MovieReviewInput,
        output_schema=MovieGenreOutput,
        instructions="Classify the movie review into its primary genre. Be accurate and provide reasoning.",
        use_chain_of_thought=True,
    )


def _create_training_set() -> List[dspy.Example]:
    """Create validated training examples."""
    _display_training_explanation()

    # Use 40 examples for training
    train_examples = TRAINING_DATASET[:40]
    trainset = []

    for ex in train_examples:
        trainset.append(
            create_dspy_example(
                MovieReviewInput,
                MovieGenreOutput,
                {"review": ex["review"]},
                {
                    "genre": ex["genre"],
                    "confidence": 0.85,
                    "reasoning": f"The review shows typical {ex['genre']} characteristics.",
                },
            )
        )

    display_success(f"Created {len(trainset)} validated training examples")
    return trainset


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


def _run_optimization(
    module: DSPyAtomicModule,
    trainset: List[dspy.Example],
) -> DSPyAtomicModule:
    """Run DSPy optimization on the bridge module."""
    display_step_header("Step 3.4: DSPy Optimization (With Schema Awareness)")

    def typed_genre_match(example, prediction, trace=None):
        """Metric that works with typed outputs."""
        pred_genre = str(prediction.genre).lower().strip()
        expected_genre = str(example.genre).lower().strip()
        return pred_genre == expected_genre

    with create_progress_context(f"[green]Running optimization ({len(trainset)} training examples)...") as progress:
        task = progress.add_task("Optimizing...", total=None)

        optimizer = dspy.BootstrapFewShot(
            metric=typed_genre_match,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            max_rounds=1,
        )
        optimized_module = optimizer.compile(module, trainset=trainset)
        progress.remove_task(task)

    display_success("Optimization complete!")
    return optimized_module


def _capture_optimized_prompt(
    lm: dspy.LM,
    optimized_module: DSPyAtomicModule,
) -> str:
    """Capture the optimized prompt."""
    display_step_header("Step 3.5: The Optimized Prompt (Exposed!)")

    with dspy.context(lm=lm):
        _ = optimized_module(review=TEST_DATASET[0]["review"])

    prompt_str = ""
    if lm.history:
        last_call = lm.history[-1]
        optimized_prompt = last_call.get("messages", [{}])
        prompt_str = json.dumps(optimized_prompt, indent=2)

        truncated = prompt_str[:2500] + ("..." if len(prompt_str) > 2500 else "")
        content = "[dim]This is what DSPy + Atomic Agents sends to the LLM:[/dim]\n\n" + truncated
        display_panel(content, "Final Optimized Prompt", "green")

    return prompt_str


# =============================================================================
# EVALUATION HELPERS
# =============================================================================


def _evaluate_module(
    optimized_module: DSPyAtomicModule,
) -> Tuple[EvalResult, List[Dict[str, Any]]]:
    """Evaluate the optimized module on test set."""
    display_step_header("Step 3.6: Evaluation with Type-Safe Outputs")

    predictions = []
    start_time = time.time()

    with create_progress_context("[green]Running predictions...") as progress:
        task = progress.add_task("Predicting...", total=len(TEST_DATASET))

        for test_ex in TEST_DATASET:
            prediction = _get_single_prediction(optimized_module, test_ex)
            predictions.append(prediction)
            progress.advance(task)

    elapsed = time.time() - start_time
    eval_result = evaluate_predictions(predictions, TEST_DATASET)
    eval_result.avg_time = elapsed / len(TEST_DATASET)

    return eval_result, predictions


def _get_single_prediction(
    module: DSPyAtomicModule,
    test_example: Dict[str, str],
) -> Dict[str, Any]:
    """Get a single validated prediction."""
    try:
        # Use run_validated to get Pydantic-validated output
        validated_result = module.run_validated(review=test_example["review"])

        return {
            "genre": validated_result.genre,  # Guaranteed Literal type!
            "confidence": validated_result.confidence,  # Guaranteed 0-1 float!
            "reasoning": validated_result.reasoning,
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


def _display_stage_results(eval_result: EvalResult) -> None:
    """Display stage 3 results and analysis."""
    display_step_header("Step 3.7: The Combined Benefits")

    content = """[green]✓ DSPy BENEFITS:[/green]
• Automatic few-shot example selection
• Optimized prompt instructions
• Chain-of-thought reasoning
• Measurable improvement through metrics

[green]✓ ATOMIC AGENTS BENEFITS:[/green]
• genre is Literal['action','comedy',...] - always valid
• confidence is float with ge=0, le=1 - always in range
• Full IDE autocomplete and type checking
• Pydantic validation catches any LLM mistakes

[yellow]COMBINED:[/yellow] Optimized prompts + Guaranteed structure!"""

    display_panel(content, "The Best of Both Worlds", "green")
    display_results_table(eval_result, "Stage 3 Results", show_confidence=True)
