"""CLI entrypoint for running the 10,000-concurrency RAG load benchmark."""

from __future__ import annotations

import argparse

from rag_platform.benchmark import run_benchmark_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10,000-concurrency Enterprise RAG benchmark.")
    parser.add_argument("--requests", type=int, default=10000, help="Total concurrent requests")
    args = parser.parse_args()

    print(f"Launching {args.requests:,} concurrent RAG requests...")
    report = run_benchmark_sync(total_requests=args.requests)
    s = report["summary"]
    ol = report["overall_latency_ms"]
    print("\n=== 10k-Concurrency Benchmark Results ===")
    print(
        f"Completed: {s['succeeded']:,}/{report['config']['total_requests']:,} "
        f"({s['success_rate']*100:.2f}%) in {s['wall_time_seconds']:.3f}s "
        f"-> {s['achieved_rps']:,.1f} RPS"
    )
    print(
        f"Latency (ms): P50={ol['p50']:.3f} | P95={ol['p95']:.3f} | "
        f"P99={ol['p99']:.3f} | Max={ol['max']:.3f}"
    )
    print(
        f"Cache + Singleflight Offload: {s['effective_cache_and_coalesce_rate']*100:.2f}% | "
        f"Peak Backend Concurrency: {s['peak_backend_concurrency']}"
    )
    for tier, info in report["tier_breakdown"].items():
        print(
            f"  - {tier:22s}: {info['count']:5,d} ({info['share_pct']:5.2f}%) | "
            f"P50={info['latency_ms']['p50']:.3f}ms P99={info['latency_ms']['p99']:.3f}ms"
        )


if __name__ == "__main__":
    main()
