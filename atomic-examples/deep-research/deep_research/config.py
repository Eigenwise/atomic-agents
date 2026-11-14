"""Configuration for the deep-research example."""

import os
from dataclasses import dataclass
from typing import Optional


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")
    return api_key


def get_searxng_base_url() -> str:
    return os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")


def get_searxng_api_key() -> Optional[str]:
    return os.getenv("SEARXNG_API_KEY")


@dataclass
class ChatConfig:
    """Model and connectivity settings. Not meant to be instantiated."""

    api_key: str = get_api_key()
    model: str = "gpt-5.6-terra"
    reasoning_effort: str = "low"
    searxng_base_url: str = get_searxng_base_url()
    searxng_api_key: Optional[str] = get_searxng_api_key()

    def __init__(self):
        raise TypeError("ChatConfig is not meant to be instantiated")


@dataclass
class ResearchBudget:
    """Hard and soft limits on the research loop.

    These are the knobs that decide how *deep* the deep research goes.
    The orchestrator respects each independently: you can't escape the
    loop by satisfying only one.
    """

    # Breadth — how many sub-topics the planner produces.
    num_sub_topics: int = 4

    # Depth — max iterations *per* sub-topic. The reflector can stop earlier.
    max_depth_per_sub_topic: int = 2

    # Per-search and per-iteration throttles.
    search_results_per_query: int = 5
    scrape_top_n_per_iteration: int = 3

    # Hard cap across the whole run, in case an agent goes rogue or a loop bug slips through.
    hard_call_cap: int = 80

    # Max characters of scraped content passed to the extractor. A handful
    # of claims only needs a few thousand chars of context, and some pages
    # (long Wikipedia articles, badly-parsed PDFs) can blow the model's
    # context window otherwise.
    max_extractor_content_chars: int = 12_000

    def __init__(self):
        raise TypeError("ResearchBudget is not meant to be instantiated")
