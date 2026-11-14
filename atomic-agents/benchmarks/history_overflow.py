"""Compare public add_message overflow before/after a revision.

Run from the repository root with the core package installed:
    python atomic-agents/benchmarks/history_overflow.py --before-ref <git-ref>
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types

from atomic_agents import BaseIOSchema
from atomic_agents.context import ChatHistory


class Entry(BaseIOSchema):
    """Benchmark message with stable serializable content."""

    number: int


def previous_history(ref):
    path = "atomic-agents/atomic_agents/context/chat_history.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    module = types.ModuleType("_history_overflow_before")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return module.ChatHistory


def compare(before, count, capacity, pairs):
    seed = ChatHistory()
    seed.current_turn_id = "benchmark-turn"
    for number in range(count):
        seed.add_message("user", Entry(number=number))
    added = Entry(number=count)
    samples = {"before_ms": [], "after_ms": []}
    expected_dump = None

    def run(cls):
        history = cls(max_messages=capacity)
        history.history = list(seed.history)
        history.current_turn_id = seed.current_turn_id
        original_list = history.history
        start = time.perf_counter_ns()
        history.add_message("assistant", added)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        assert history.history is original_list
        return elapsed, history.dump(), history.get_history()

    for pair in range(pairs + 2):
        order = [("before_ms", before), ("after_ms", ChatHistory)]
        if pair % 2:
            order.reverse()
        outputs = []
        for name, cls in order:
            elapsed, dumped, messages = run(cls)
            outputs.append((dumped, messages))
            if pair >= 2:
                samples[name].append(elapsed)
        assert outputs[0] == outputs[1]
        if expected_dump is None:
            expected_dump = outputs[0][0]
        assert outputs[0][0] == expected_dump

    return {
        "messages_before_add": count,
        "capacity": capacity,
        "pairs": pairs,
        "before_median_ms": statistics.median(samples["before_ms"]),
        "after_median_ms": statistics.median(samples["after_ms"]),
        "output_sha256": hashlib.sha256(expected_dump.encode()).hexdigest(),
        **samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-ref", required=True)
    parser.add_argument("--pairs", type=int, default=7)
    args = parser.parse_args()
    before = previous_history(args.before_ref)
    cases = [(100_000, 32), (100_000, 0), (1000, 1000), (20, 20), (20, None)]
    print(json.dumps([compare(before, count, capacity, args.pairs) for count, capacity in cases], indent=2))


if __name__ == "__main__":
    main()
