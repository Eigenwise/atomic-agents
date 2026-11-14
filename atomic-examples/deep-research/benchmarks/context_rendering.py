"""Compare complete context rendering with a selected repository revision.

From the repository root, with atomic-agents installed:
python atomic-examples/deep-research/benchmarks/context_rendering.py --baseline-ref BASE_REF
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_research.context_providers import ResearchStateProvider  # noqa: E402
from deep_research.state import Learning, ResearchState, Source  # noqa: E402


def measure(provider, expected, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        actual = provider.get_info()
    milliseconds = (time.perf_counter() - start) * 1000 / iterations
    assert actual == expected, "complete rendered context changed"
    return milliseconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    path = "atomic-examples/deep-research/deep_research/context_providers.py"
    source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:{path}"], cwd=root, text=True)
    namespace = {"__name__": "context_rendering_baseline"}
    exec(compile(source, path, "exec"), namespace)
    previous = namespace["ResearchStateProvider"]
    results = []
    for name, count, topics, iterations in [("many-topics", 6000, 600, 3), ("one-topic", 6000, 1, 5), ("short", 30, 5, 200)]:
        state = ResearchState(
            question="Synthetic context rendering comparison",
            sources=[Source("S1", "https://example.test/source", "Example source")],
            learnings=[Learning(f"Finding {index}: stable text", "S1", f"Topic {index % topics}") for index in range(count)],
        )
        providers = {"before": previous("Research", state), "after": ResearchStateProvider("Research", state)}
        expected = providers["before"].get_info()
        assert providers["after"].get_info() == expected
        for provider in providers.values():
            for _ in range(2):
                measure(provider, expected, iterations)
        timings = {"before": [], "after": []}
        for sample in range(7):
            for variant in ["before", "after"] if sample % 2 == 0 else ["after", "before"]:
                timings[variant].append(measure(providers[variant], expected, iterations))
        results.append(
            {
                "name": name,
                "learnings": count,
                "topics": topics,
                "iterations_per_sample": iterations,
                "characters": len(expected),
                "sha256": hashlib.sha256(expected.encode()).hexdigest(),
                "milliseconds_per_render": timings,
                "median_milliseconds": {name: statistics.median(samples) for name, samples in timings.items()},
            }
        )
    print(json.dumps({"python": platform.python_version(), "baseline_ref": args.baseline_ref, "results": results}, indent=2))


if __name__ == "__main__":
    main()
