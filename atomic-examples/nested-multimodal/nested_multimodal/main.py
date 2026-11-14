"""
Nested Multimodal Example
=========================

Demonstrates that Atomic Agents correctly handles multimodal content (images,
PDFs) inside nested Pydantic schemas — not just at the top level.

This example exercises the fixes for:
  - GitHub #208: nested Pydantic model + top-level multimodal → TypeError
  - GitHub #141: multimodal inside nested schemas invisible to ChatHistory
"""

import json
import os
from typing import List

import instructor
import openai
from dotenv import load_dotenv
from pydantic import Field

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

load_dotenv()

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )


# ---------------------------------------------------------------------------
# Schemas — nested multimodal content
# ---------------------------------------------------------------------------
class AnalysisContext(BaseIOSchema):
    """Additional context for the analysis request."""

    focus_area: str = Field(..., description="What aspect to focus the analysis on")
    detail_level: str = Field(..., description="How detailed the analysis should be (brief / detailed)")


class ImageWithContext(BaseIOSchema):
    """An image wrapped in a nested schema together with metadata."""

    image: instructor.Image = Field(..., description="The image to analyze")
    label: str = Field(..., description="A short human-readable label for this image")


class AnalysisInput(BaseIOSchema):
    """Input schema that combines nested multimodal content with a nested Pydantic context object."""

    documents: List[ImageWithContext] = Field(..., description="Images to analyze, each with a label")
    context: AnalysisContext = Field(..., description="Analysis context and preferences")
    instruction: str = Field(..., description="What the agent should do with the images")


class AnalysisOutput(BaseIOSchema):
    """Structured output from the image analysis."""

    summary: str = Field(..., description="Overall summary of all analyzed images")
    per_image: List[str] = Field(..., description="One description per image, in the same order as the input")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = AtomicAgent[AnalysisInput, AnalysisOutput](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=API_KEY)),
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an image analysis assistant.",
                "You receive images wrapped inside document objects, each with a label.",
                "You also receive a context object that tells you what to focus on.",
            ],
            steps=[
                "1. Look at each image and its label.",
                "2. Analyze according to the focus_area and detail_level in the context.",
                "3. Write a per-image description and an overall summary.",
            ],
            output_instructions=[
                "Return a summary covering all images and a list of per-image descriptions.",
            ],
        ),
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def verify_history_format(agent_instance: AtomicAgent) -> None:
    """Print the serialized chat history so we can confirm the fix works."""
    history = agent_instance.history.get_history()
    print("\n--- Chat history entries ---")
    for i, entry in enumerate(history):
        role = entry["role"]
        content = entry["content"]
        if isinstance(content, list):
            text_parts = [json.loads(c) if isinstance(c, str) else type(c).__name__ for c in content]
            print(f"  [{i}] role={role}  content (list with {len(content)} items):")
            for j, part in enumerate(text_parts):
                print(f"        [{j}] {part}")
        else:
            print(f"  [{i}] role={role}  content={content[:120]}...")
    print("--- end ---\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Nested Multimodal Example ===\n")

    # Build the input — images nested inside ImageWithContext schemas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(os.path.dirname(script_dir), "test_images")
    image_path = os.path.join(test_images_dir, "nutrition_label_1.png")

    analysis_input = AnalysisInput(
        documents=[
            ImageWithContext(
                image=instructor.Image.from_path(image_path),
                label="Nutrition label photo",
            ),
        ],
        context=AnalysisContext(
            focus_area="nutritional content",
            detail_level="brief",
        ),
        instruction="Describe what you see in each image, paying attention to the focus area.",
    )

    # --- Verify the history format (no LLM call yet) -----------------------
    print("Step 1: Adding message to history and verifying serialization...\n")
    agent.history.add_message("user", analysis_input)
    verify_history_format(agent)

    # Confirm the nested Image was extracted and the nested AnalysisContext
    # was serialized properly (this is what Issues #208 / #141 broke).
    history = agent.history.get_history()
    assert isinstance(history[0]["content"], list), "Content should be a multimodal list"
    json_part = json.loads(history[0]["content"][0])
    assert "context" in json_part, "Nested AnalysisContext should be in the JSON"
    assert json_part["context"]["focus_area"] == "nutritional content"
    assert any(
        isinstance(item, instructor.Image) for item in history[0]["content"]
    ), "Image should be extracted into the content list"
    print("Serialization OK — nested context preserved, nested image extracted.\n")

    # Reset history before the real run (the agent adds messages internally)
    agent.reset_history()

    # --- End-to-end LLM call ------------------------------------------------
    print("Step 2: Running the agent end-to-end...\n")
    result = agent.run(analysis_input)

    print("Agent response:")
    print(f"  Summary : {result.summary}")
    for i, desc in enumerate(result.per_image, 1):
        print(f"  Image {i}: {desc}")

    # Show the full history after the run
    verify_history_format(agent)
    print("Done — nested multimodal schemas work end-to-end!")


if __name__ == "__main__":
    main()
