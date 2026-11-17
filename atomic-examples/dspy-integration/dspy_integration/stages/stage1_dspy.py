"""
Stage 1: Raw DSPy with Typed Signatures.

This module demonstrates DSPy's capabilities at their best:
- Typed signatures with Literal constraints
- Automatic prompt optimization via BootstrapFewShot
- Chain-of-thought reasoning

Limitations shown:
- No Pydantic validation ecosystem
- Less integration with structured output tools
- Type enforcement is DSPy-specific, not Python runtime

Design: Single function entry point, internal helpers follow SRP.
"""

import json
import time
from typing import Any, Dict, List, Tuple

import dspy

from dspy_integration.domain.models import (
    GENRES,
    GenreType,
    EvalResult,
)
from dspy_integration.domain.datasets import TRAINING_DATASET, TEST_DATASET
from dspy_integration.domain.evaluation import evaluate_predictions
from dspy_integration.presentation.console import (
    console,
    display_stage_header,
    display_panel,
    display_code,
    display_step_header,
    display_success,
    display_tree,
    display_results_table,
    create_progress_context,
)


# =============================================================================
# DSPY SIGNATURE DEFINITION
# =============================================================================


class MovieGenreSignature(dspy.Signature):
    """
    Classify a movie review into its primary genre based on the review text.

    Consider the overall focus and tone of the review, not just individual keywords.
    A review mentioning 'explosions' might be a drama if the focus is on characters.
    A 'scary' movie might be a comedy if played for laughs.
    """

    review: str = dspy.InputField(desc="The movie review text to classify")
    genre: GenreType = dspy.OutputField(desc="The primary genre: action, comedy, drama, horror, sci-fi, or romance")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")


# =============================================================================
# CODE EXAMPLES FOR DISPLAY
# =============================================================================


SIGNATURE_CODE_EXAMPLE = '''from typing import Literal

# DSPy Signature WITH proper type constraints
class MovieGenreSignature(dspy.Signature):
    """Classify a movie review into its primary genre."""

    review: str = dspy.InputField(desc="The movie review text")

    # Literal type constrains output to valid genres only!
    genre: Literal["action", "comedy", "drama", "horror", "sci-fi", "romance"] = \\
        dspy.OutputField(desc="The primary genre")

    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation")

# DSPy enforces the Literal constraint - no more "dramedy" or "thriller"!
classify = dspy.ChainOfThought(MovieGenreSignature)'''


# =============================================================================
# MAIN STAGE FUNCTION
# =============================================================================


def run_stage1_raw_dspy(api_key: str) -> Tuple[EvalResult, Dict[str, Any]]:
    """
    Run Stage 1: Raw DSPy demonstration.

    This demonstrates DSPy at its best with proper typed signatures.

    Args:
        api_key: OpenAI API key

    Returns:
        Tuple of (evaluation results, behind-the-scenes data)
    """
    display_stage_header("STAGE 1: Raw DSPy (Properly Implemented)", "blue")
    _display_stage_overview()

    # Configure DSPy
    lm = dspy.LM("openai/gpt-5.6-terra", api_key=api_key)
    dspy.configure(lm=lm)

    # Step 1: Show signature
    _display_signature_explanation()

    # Step 2: Create classifier and show unoptimized prompt
    classify = dspy.ChainOfThought(MovieGenreSignature)
    unoptimized_prompt = _capture_unoptimized_prompt(lm, classify)

    # Step 3: Explain optimization
    _display_optimization_explanation()

    # Step 4: Run optimization
    optimized_classify = _run_optimization(lm, classify)

    # Step 5: Show optimized prompt
    optimized_prompt = _capture_optimized_prompt(lm, optimized_classify)

    # Step 6: Show selected demos
    _display_selected_demos(optimized_classify)

    # Step 7: Evaluate
    eval_result, predictions = _evaluate_model(optimized_classify)

    # Step 8: Display results
    _display_stage_results(eval_result, predictions)

    behind_scenes = _create_behind_scenes_data(unoptimized_prompt, optimized_prompt, optimized_classify)

    return eval_result, behind_scenes


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def _display_stage_overview() -> None:
    """Display stage 1 overview panel."""
    content = """[green]DSPy STRENGTHS:[/green]
• Typed signatures with Literal constraints (genre MUST be valid)
• Automatic prompt optimization via BootstrapFewShot
• Chain-of-thought reasoning for complex decisions
• Systematic few-shot example selection

[yellow]LIMITATIONS vs Atomic Agents:[/yellow]
• No Pydantic ecosystem (validators, serializers, etc.)
• Less integration with structured output tools like Instructor
• Type hints are enforced by DSPy, not Python runtime"""

    display_panel(content, "Stage 1 Overview", "blue")


def _display_signature_explanation() -> None:
    """Display explanation of DSPy typed signatures."""
    display_step_header("Step 1.1: Define Typed DSPy Signature")
    console.print("DSPy supports class-based signatures with Python type hints:\n")
    display_code(SIGNATURE_CODE_EXAMPLE)


def _display_optimization_explanation() -> None:
    """Display explanation of how DSPy optimization works."""
    display_step_header("Step 1.3: DSPy Optimization (BootstrapFewShot)")

    content = """[cyan]What BootstrapFewShot does:[/cyan]

1. Takes your labeled training examples
2. Runs the LLM on each to generate 'traces' (reasoning chains)
3. Filters traces that produce correct answers
4. Selects the best traces as few-shot demonstrations
5. Injects these into future prompts automatically

[yellow]Key insight:[/yellow] DSPy doesn't just use your examples verbatim.
It generates NEW reasoning and picks what actually works!"""

    display_panel(content, "How DSPy Optimization Works", "cyan")


# =============================================================================
# PROMPT CAPTURE HELPERS
# =============================================================================


def _capture_unoptimized_prompt(
    lm: dspy.LM,
    classify: dspy.Module,
) -> List[Dict[str, Any]]:
    """Capture the unoptimized prompt from DSPy."""
    display_step_header("Step 1.2: Unoptimized Prompt (What DSPy Generates)")

    with dspy.context(lm=lm):
        _ = classify(review=TRAINING_DATASET[0]["review"])

    unoptimized_prompt = []
    if lm.history:
        last_call = lm.history[-1]
        unoptimized_prompt = last_call.get("messages", [{}])

        content = (
            "[dim]Notice how DSPy includes the Literal type constraint in the prompt:[/dim]\n\n"
            + json.dumps(unoptimized_prompt, indent=2)[:2000]
            + "..."
        )
        display_panel(content, "Unoptimized DSPy Prompt (With Type Constraints)", "yellow")

    return unoptimized_prompt


def _capture_optimized_prompt(
    lm: dspy.LM,
    optimized_classify: dspy.Module,
) -> List[Dict[str, Any]]:
    """Capture the optimized prompt from DSPy."""
    display_step_header("Step 1.4: Optimized Prompt (After DSPy Magic)")

    with dspy.context(lm=lm):
        _ = optimized_classify(review=TEST_DATASET[0]["review"])

    optimized_prompt = []
    if lm.history:
        last_call = lm.history[-1]
        optimized_prompt = last_call.get("messages", [{}])

        prompt_str = json.dumps(optimized_prompt, indent=2)
        truncated = prompt_str[:3500] + ("..." if len(prompt_str) > 3500 else "")

        content = "[dim]Notice the auto-selected few-shot examples with reasoning:[/dim]\n\n" + truncated
        display_panel(content, "Optimized DSPy Prompt (With Auto-Selected Examples)", "green")

    return optimized_prompt


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


def _run_optimization(lm: dspy.LM, classify: dspy.Module) -> dspy.Module:
    """Run DSPy optimization with BootstrapFewShot."""
    # Prepare training set (first 30 examples)
    train_examples = TRAINING_DATASET[:30]
    trainset = [
        dspy.Example(
            review=ex["review"],
            genre=ex["genre"],
            confidence=0.85,
            reasoning=f"This review demonstrates typical {ex['genre']} characteristics.",
        ).with_inputs("review")
        for ex in train_examples
    ]

    def genre_match(example, prediction, trace=None):
        """Metric for optimization - checks if genre matches."""
        pred_genre = str(prediction.genre).lower().strip()
        expected_genre = str(example.genre).lower().strip()
        return pred_genre == expected_genre

    with create_progress_context("[cyan]Running DSPy optimization (30 training examples)...") as progress:
        task = progress.add_task("Optimizing...", total=None)

        optimizer = dspy.BootstrapFewShot(
            metric=genre_match,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            max_rounds=1,
        )
        optimized_classify = optimizer.compile(classify, trainset=trainset)
        progress.remove_task(task)

    display_success("Optimization complete!")
    return optimized_classify


def _display_selected_demos(optimized_classify: dspy.Module) -> None:
    """Display the few-shot examples DSPy selected."""
    display_step_header("Step 1.5: Few-Shot Examples DSPy Selected")

    if hasattr(optimized_classify, "demos") and optimized_classify.demos:
        items = []
        for i, demo in enumerate(optimized_classify.demos[:4]):
            review_text = str(getattr(demo, "review", "N/A"))[:70]
            genre = getattr(demo, "genre", "N/A")
            reasoning = str(getattr(demo, "reasoning", ""))[:80]

            items.append(
                {
                    "title": f"Example {i + 1}",
                    "children": [
                        f"Review: {review_text}...",
                        f"Genre: [green]{genre}[/green]",
                        f"Reasoning: [dim]{reasoning}...[/dim]",
                    ],
                }
            )

        display_tree("Selected Demonstrations", items)
    else:
        console.print("[dim]Demo inspection not available for this predictor type[/dim]")


# =============================================================================
# EVALUATION HELPERS
# =============================================================================


def _evaluate_model(
    optimized_classify: dspy.Module,
) -> Tuple[EvalResult, List[Dict[str, Any]]]:
    """Evaluate the optimized model on test set."""
    display_step_header(f"Step 1.6: Evaluation on Test Set ({len(TEST_DATASET)} challenging examples)")

    predictions = []
    start_time = time.time()

    with create_progress_context("[cyan]Running predictions...") as progress:
        task = progress.add_task("Predicting...", total=len(TEST_DATASET))

        for test_ex in TEST_DATASET:
            prediction = _get_single_prediction(optimized_classify, test_ex)
            predictions.append(prediction)
            progress.advance(task)

    elapsed = time.time() - start_time
    eval_result = evaluate_predictions(predictions, TEST_DATASET)
    eval_result.avg_time = elapsed / len(TEST_DATASET)

    return eval_result, predictions


def _get_single_prediction(
    classifier: dspy.Module,
    test_example: Dict[str, str],
) -> Dict[str, Any]:
    """Get a single prediction from the classifier."""
    try:
        result = classifier(review=test_example["review"])
        genre_val = str(result.genre).strip().lower()

        # Validate genre
        if genre_val not in GENRES:
            genre_val = "error"

        return {
            "genre": genre_val,
            "confidence": float(result.confidence) if hasattr(result, "confidence") else 0.5,
            "reasoning": str(result.reasoning) if hasattr(result, "reasoning") else "N/A",
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
    """Display stage 1 results and analysis."""
    display_step_header("Step 1.7: Results")

    # Count invalid genres
    invalid_genres = [p["genre"] for p in predictions if p["genre"] not in GENRES]

    content = f"""[green]DSPy TYPED SIGNATURE BENEFITS:[/green]
• Genre constrained to valid options (invalid outputs: {len(invalid_genres)})
• Automatic few-shot example selection
• Chain-of-thought reasoning included

[yellow]REMAINING LIMITATIONS:[/yellow]
• No Pydantic validation ecosystem
• Confidence not guaranteed to be 0-1 (no ge/le constraints)
• Can't use Instructor's retry mechanisms
• Type enforcement is DSPy-specific, not Python-native"""

    display_panel(content, "DSPy Typed Signatures Assessment", "blue")
    display_results_table(eval_result, "Stage 1 Results")


def _create_behind_scenes_data(
    unoptimized_prompt: List[Dict[str, Any]],
    optimized_prompt: List[Dict[str, Any]],
    optimized_classify: dspy.Module,
) -> Dict[str, Any]:
    """Create behind-the-scenes data for comparison."""
    return {
        "unoptimized_prompt_sample": str(unoptimized_prompt)[:500],
        "optimized_prompt_sample": str(optimized_prompt)[:500],
        "num_demos_selected": (len(optimized_classify.demos) if hasattr(optimized_classify, "demos") else "N/A"),
        "training_examples": 30,
    }
