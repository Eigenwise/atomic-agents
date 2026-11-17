"""
DSPy + Atomic Agents Integration: A Comprehensive Didactic Example.

This example teaches you WHY combining DSPy with Atomic Agents is powerful
by walking through three stages with a large, challenging benchmark.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py (Orchestrator)                         │
│                    - Entry point, coordinates all stages                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  stages/                          │  domain/                                │
│  ├── stage1_dspy.py              │  ├── models.py (schemas, types)        │
│  ├── stage2_atomic.py            │  ├── datasets.py (train/test data)     │
│  └── stage3_combined.py          │  └── evaluation.py (metrics)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  presentation/                    │  bridge.py                             │
│  └── console.py (Rich UI)        │  (DSPy ↔ Atomic Agents)               │
└─────────────────────────────────────────────────────────────────────────────┘

Run: uv run python -m dspy_integration.main

Clean Architecture Principles Applied:
- Separation of Concerns: Each module has a single responsibility
- Dependency Inversion: High-level modules don't depend on low-level details
- Single Responsibility: Each function/class has one reason to change
- Open/Closed: Easy to extend (add new stages) without modifying existing code
"""

import os
import random
import traceback

from dotenv import load_dotenv

from dspy_integration.domain.models import EvalResult
from dspy_integration.domain.datasets import TRAINING_DATASET, TEST_DATASET
from dspy_integration.stages import (
    run_stage1_raw_dspy,
    run_stage2_raw_atomic_agents,
    run_stage3_combined,
)
from dspy_integration.presentation.console import (
    console,
    display_welcome,
    display_comparison_table,
    display_takeaways,
    display_decision_guide,
    display_stage_header,
)

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
random.seed(42)


# =============================================================================
# ORCHESTRATION
# =============================================================================


def run_all_stages(api_key: str) -> None:
    """
    Run all three demonstration stages.

    This is the main orchestration function that coordinates
    the execution of all stages and displays the final comparison.

    Args:
        api_key: OpenAI API key for LLM access
    """
    # Stage 1: Raw DSPy
    stage1_result, _ = run_stage1_raw_dspy(api_key)
    console.print("\n")

    # Stage 2: Raw Atomic Agents
    stage2_result, _ = run_stage2_raw_atomic_agents(api_key)
    console.print("\n")

    # Stage 3: Combined approach
    stage3_result, _ = run_stage3_combined(api_key)
    console.print("\n")

    # Final comparison
    show_final_comparison(stage1_result, stage2_result, stage3_result)


def show_final_comparison(
    stage1_result: EvalResult,
    stage2_result: EvalResult,
    stage3_result: EvalResult,
) -> None:
    """
    Display side-by-side comparison of all three approaches.

    This provides the key takeaway - showing why combining
    DSPy with Atomic Agents gives the best results.
    """
    display_stage_header("FINAL COMPARISON", "yellow")
    display_comparison_table(stage1_result, stage2_result, stage3_result)
    display_takeaways()
    display_decision_guide()


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Main entry point for the demonstration.

    Responsibilities:
    - Display welcome message
    - Validate API key
    - Run all stages
    - Handle errors gracefully
    """
    display_welcome(
        title="DSPy + Atomic Agents: A Comprehensive Didactic Example",
        subtitle=(
            "This example teaches you WHY combining these frameworks is powerful\n"
            "by walking through three stages with full transparency."
        ),
        details=(
            f"Large benchmark: {len(TRAINING_DATASET)} training examples, "
            f"{len(TEST_DATASET)} challenging test cases\n"
            "We'll expose the prompts, show the optimizations,\n"
            "and compare measurable results."
        ),
    )

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable required[/red]")
        return

    # Display configuration
    console.print("\n[dim]Using model: gpt-5.6-terra[/dim]")
    console.print(f"[dim]Training set: {len(TRAINING_DATASET)} examples (balanced across 6 genres)[/dim]")
    console.print(f"[dim]Test set: {len(TEST_DATASET)} challenging examples " "(sarcasm, multi-genre, etc.)[/dim]\n")

    # Run demonstration
    try:
        run_all_stages(api_key)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
