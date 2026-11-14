"""Configuration module for FastAPI Atomic Agents example."""

import os


def get_api_key() -> str:
    """Get OpenAI API key from environment variables.

    Returns:
        str: OpenAI API key

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your environment before running the application."
        )
    return api_key


# Constants
DEFAULT_SESSION_ID = "default"
MODEL_NAME = "gpt-5.6-terra"
NUM_SUGGESTED_QUESTIONS = 3
