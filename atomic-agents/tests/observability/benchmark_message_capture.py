"""Compare message capture with fresh processes and two existing source trees."""

import argparse
import gc
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time
import tracemalloc


def worker(source, shape, iterations):
    sys.path.insert(0, str(pathlib.Path(source) / "atomic-agents"))
    from atomic_agents.observability import TracingConfig
    from atomic_agents.observability.message_payloads import write_messages
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    if shape == "history":
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"{i:05d}: " + ('Context 漢字 "quoted" line\n' * 20)}
            for i in range(10000)
        ]
    elif shape == "small":
        messages = [{"role": "user", "content": f"{i}: normal context " * 5} for i in range(4)]
    else:
        messages = [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]
    input_bytes = json.dumps(messages, ensure_ascii=False).encode()
    input_digest = hashlib.sha256(input_bytes).hexdigest()
    input_size = len(input_bytes)
    del input_bytes
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("message-capture-benchmark")
    config = TracingConfig(capture_messages=True, max_attribute_length=4096)

    def capture():
        with tracer.start_as_current_span("chat") as span:
            write_messages(span, messages, "input", config)
        result = dict(exporter.get_finished_spans()[0].attributes)
        exporter.clear()
        return result

    capture()
    gc.collect()
    wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    for _ in range(iterations):
        result = capture()
    wall_ns, cpu_ns = time.perf_counter_ns() - wall_start, time.process_time_ns() - cpu_start
    gc.collect()
    tracemalloc.start()
    memory_result = capture()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result == memory_result
    provider.shutdown()
    print(
        json.dumps(
            {
                "shape": shape,
                "iterations": iterations,
                "messages": len(messages),
                "input_bytes": input_size,
                "input_sha256": input_digest,
                "wall_ms": wall_ns / 1e6,
                "cpu_ms": cpu_ns / 1e6,
                "peak_python_traced_bytes": peak_bytes,
                "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "result_attribute_count": len(result),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--shape", choices=["history", "small", "single"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.worker:
        if not args.source or not args.shape:
            parser.error("--worker requires --source and --shape")
        worker(args.source, args.shape, args.iterations)
        return
    if not args.baseline or not args.candidate or not args.output:
        parser.error("provide --baseline, --candidate, and --output")
    rows = []
    for shape in ["history", "small", "single"]:
        for pair in range(7):
            for variant in ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]:
                result = json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            "--source",
                            getattr(args, variant),
                            "--shape",
                            shape,
                            "--iterations",
                            str(args.iterations),
                        ],
                        text=True,
                    )
                )
                rows.append({"pair": pair, "variant": variant, **result})
        scoped = [r for r in rows if r["shape"] == shape]
        assert len({r["input_sha256"] for r in scoped}) == len({r["result_sha256"] for r in scoped}) == 1
        print(json.dumps({"completed_shape": shape}), flush=True)
    metrics = ["wall_ms", "cpu_ms", "peak_python_traced_bytes"]
    medians = {
        shape: {
            variant: {
                metric: statistics.median(r[metric] for r in rows if r["shape"] == shape and r["variant"] == variant)
                for metric in metrics
            }
            for variant in ["baseline", "candidate"]
        }
        for shape in ["history", "small", "single"]
    }
    pathlib.Path(args.output).write_text(json.dumps({"rows": rows, "medians": medians}, indent=2) + "\n")
    print(json.dumps(medians), flush=True)


if __name__ == "__main__":
    main()
