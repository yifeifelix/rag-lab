"""10,000-Concurrency enterprise RAG load benchmark & tail latency profiler."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .config import RAGConfig
from .data_gen import generate_datasets, load_golden_qa
from .pipeline import EnterpriseRAGPipeline


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
    }


def _build_10k_workload(
    pipeline: EnterpriseRAGPipeline,
    golden_qa: list[dict[str, Any]],
    total_requests: int = 10000,
    seed: int = 42,
) -> tuple[list[str], list[tuple[str, bool]]]:
    """Build realistic Zipf enterprise workload:
    - Warmup seeds: canonical queries with verified >=0.90 paraphrase variants
    - 35% Hot steady-state repeats (L1 Exact Cache)
    - 10% Uncached viral bursts (Singleflight Coalescing)
    - 30% Paraphrased variants (L2 Semantic Vector Cache)
    - 25% Long-tail unique/personalized queries (Cold Hybrid RRF + Reranker)
    """
    rng = random.Random(seed)
    answerable = [q for q in golden_qa if q.get("relevant_doc_ids")]

    # Identify canonical seeds whose paraphrase variants truly clear L2 similarity threshold
    l2_seeds: list[tuple[str, list[str]]] = []
    threshold = pipeline.config.l2_similarity_threshold
    for item in answerable:
        q_vec = pipeline.embedder.encode_single(item["question"])
        good_pvs = [
            pv
            for pv in item.get("paraphrase_variants", [])
            if float(q_vec @ pipeline.embedder.encode_single(pv)) >= threshold
        ]
        if good_pvs:
            l2_seeds.append((item["question"], good_pvs))

    if not l2_seeds:
        l2_seeds = [(answerable[0]["question"], [answerable[0]["question"]])]

    # Keep 4 separate un-warmed viral queries for Singleflight thundering-herd burst testing
    seed_questions_set = {q for q, _ in l2_seeds}
    viral_pool = [q["question"] for q in answerable if q["question"] not in seed_questions_set][:4]
    if len(viral_pool) < 2:
        viral_pool = [q["question"] for q in answerable[-4:]]

    warmup_queries = [q for q, _ in l2_seeds]

    n_hot = int(total_requests * 0.35)
    n_viral = int(total_requests * 0.10)
    n_para = int(total_requests * 0.30)
    n_tail = total_requests - (n_hot + n_viral + n_para)

    workload: list[tuple[str, bool]] = []  # (query_text, use_cache)

    # 1. Hot L1 exact repeats
    for i in range(n_hot):
        q, _ = l2_seeds[i % len(l2_seeds)]
        workload.append((q, True))

    # 2. Uncached viral spikes (arrive simultaneously -> Singleflight coalescing)
    for i in range(n_viral):
        vq = viral_pool[i % len(viral_pool)]
        workload.append((vq, True))

    # 3. Paraphrased queries (L2 semantic cache hits)
    for i in range(n_para):
        _, pvs = l2_seeds[i % len(l2_seeds)]
        pv = pvs[i % len(pvs)]
        workload.append((pv, True))

    # 4. Unique long-tail queries (personalized / uncached cold RAG execution)
    docs = pipeline.documents
    for i in range(n_tail):
        doc = docs[i % len(docs)]
        tail_q = f"Summarize compliance and architecture controls for {doc['title']} ({doc['doc_id']}) audit #{i}"
        # Personalized audit tag bypasses shared cache so cold Hybrid RRF + Reranker executes
        workload.append((tail_q, False))

    rng.shuffle(workload)
    return warmup_queries, workload


async def run_concurrency_benchmark(
    total_requests: int = 10000,
    simulated_llm_latency_ms: float = 2.0,
    max_concurrent_pipelines: int = 256,
    base_config: RAGConfig | None = None,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Execute `total_requests` concurrent RAG queries via asyncio.gather."""
    cfg = base_config or RAGConfig(
        chunking_strategy="parent_child",
        retrieval_mode="hybrid_rrf",
        use_reranker=True,
        use_cache=True,
        max_concurrent_pipelines=max_concurrent_pipelines,
        rate_limit_rps=max(50000.0, float(total_requests) * 5.0),
        rate_limit_burst=max(25000.0, float(total_requests) * 2.0),
        simulated_llm_latency_ms=simulated_llm_latency_ms,
    )
    if not cfg.corpus_path.exists() or not cfg.golden_qa_path.exists():
        generate_datasets(cfg, force=False)

    golden_qa = load_golden_qa(cfg.golden_qa_path)
    pipeline = EnterpriseRAGPipeline(config=cfg)
    warmup_queries, workload = _build_10k_workload(
        pipeline=pipeline,
        golden_qa=golden_qa,
        total_requests=total_requests,
    )

    # Seed steady-state L1/L2 caches
    for wq in warmup_queries:
        await pipeline.query(wq, use_cache=True)

    # Reset counters after warmup while preserving seeded caches
    pipeline.l1_cache.hits = 0
    pipeline.l1_cache.misses = 0
    pipeline.l2_cache.hits = 0
    pipeline.l2_cache.misses = 0
    pipeline.singleflight.reset_stats()
    pipeline.concurrency_gate.reset_stats()
    pipeline.rate_limiter.reset_stats()
    pipeline.total_queries = 0
    pipeline.cache_misses = 0

    async def _run_one(q_text: str, use_cache_flag: bool) -> tuple[str, float, bool]:
        resp = await pipeline.query(q_text, use_cache=use_cache_flag)
        ok = resp.cache_tier != "rate_limited" and (bool(resp.answer) or resp.refused)
        return resp.cache_tier, resp.latency_ms, ok

    t_start = time.perf_counter()
    results = await asyncio.gather(*(_run_one(q, uc) for q, uc in workload))
    wall_seconds = max(1e-6, time.perf_counter() - t_start)

    all_latencies: list[float] = []
    latencies_by_tier: dict[str, list[float]] = {
        "l1_exact": [],
        "l2_semantic": [],
        "singleflight_coalesced": [],
        "miss": [],
        "rate_limited": [],
    }
    succeeded = 0
    for tier, lat, ok in results:
        if ok:
            succeeded += 1
        all_latencies.append(lat)
        latencies_by_tier.setdefault(tier, []).append(lat)

    tier_counts = {k: len(v) for k, v in latencies_by_tier.items() if len(v) > 0}
    effective_cache_savings = (
        tier_counts.get("l1_exact", 0)
        + tier_counts.get("l2_semantic", 0)
        + tier_counts.get("singleflight_coalesced", 0)
    ) / total_requests

    report: dict[str, Any] = {
        "config": {
            "total_requests": total_requests,
            "chunking_strategy": cfg.chunking_strategy,
            "retrieval_mode": cfg.retrieval_mode,
            "use_reranker": cfg.use_reranker,
            "max_concurrent_pipelines": cfg.max_concurrent_pipelines,
            "l2_similarity_threshold": cfg.l2_similarity_threshold,
            "simulated_llm_latency_ms": cfg.simulated_llm_latency_ms,
        },
        "summary": {
            "wall_time_seconds": round(wall_seconds, 3),
            "achieved_rps": round(total_requests / wall_seconds, 1),
            "succeeded": succeeded,
            "failed": total_requests - succeeded,
            "success_rate": round(succeeded / total_requests, 6),
            "peak_backend_concurrency": pipeline.concurrency_gate.peak_concurrency,
            "effective_cache_and_coalesce_rate": round(effective_cache_savings, 4),
        },
        "overall_latency_ms": _percentiles(all_latencies),
        "tier_breakdown": {
            tier: {
                "count": len(lats),
                "share_pct": round(100.0 * len(lats) / total_requests, 2),
                "latency_ms": _percentiles(lats),
            }
            for tier, lats in latencies_by_tier.items()
            if lats
        },
    }

    if save_reports and total_requests >= 1000:
        reports_dir = Path(cfg.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "benchmark_10k.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        (reports_dir / "benchmark_10k_report.md").write_text(
            _render_benchmark_markdown(report),
            encoding="utf-8",
        )

    return report


def _render_benchmark_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    ol = report["overall_latency_ms"]
    cfg = report["config"]
    lines = [
        f"# {cfg['total_requests']:,}-Concurrency Enterprise RAG Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- **Total Concurrent Requests**: `{cfg['total_requests']:,}`",
        f"- **Succeeded / Failed**: `{s['succeeded']:,} / {s['failed']:,}` (`{s['success_rate'] * 100:.2f}%` success rate)",
        f"- **Wall-Clock Duration**: `{s['wall_time_seconds']:.3f} s`",
        f"- **Achieved Throughput**: `{s['achieved_rps']:,.1f} RPS`",
        f"- **Peak Backend Concurrency (Gate Cap={cfg['max_concurrent_pipelines']})**: `{s['peak_backend_concurrency']}`",
        f"- **Cache + Singleflight Offload Rate**: `{s['effective_cache_and_coalesce_rate'] * 100:.2f}%`",
        "",
        "## Overall Latency Distribution (ms)",
        "",
        f"- **P50**: `{ol['p50']:.3f} ms` | **P95**: `{ol['p95']:.3f} ms` | **P99**: `{ol['p99']:.3f} ms` | **Max**: `{ol['max']:.3f} ms`",
        "",
        "## Execution Tier Breakdown",
        "",
        "| Tier | Count | Traffic Share | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, info in report["tier_breakdown"].items():
        lat = info["latency_ms"]
        lines.append(
            f"| `{tier}` | {info['count']:,} | {info['share_pct']:.2f}% | "
            f"{lat['p50']:.3f} | {lat['p95']:.3f} | {lat['p99']:.3f} | {lat['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_benchmark_sync(total_requests: int = 10000) -> dict[str, Any]:
    """Synchronous CLI wrapper."""
    return asyncio.run(run_concurrency_benchmark(total_requests=total_requests, save_reports=True))
