"""Data Science RAG evaluation framework: retrieval, lexical proxies, bootstrap CIs & ablation runner."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import RAGConfig
from .data_gen import generate_datasets, load_corpus_documents, load_golden_qa
from .pipeline import EnterpriseRAGPipeline, RAGResponse


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute mean and non-parametric 95% bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return {"mean": round(mean_val, 4), "ci_low": round(mean_val, 4), "ci_high": round(mean_val, 4), "n": 1}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return {
        "mean": round(mean_val, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "n": int(arr.size),
    }


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute paired mean improvement and fraction of paired bootstrap mean deltas <= 0 (not a calibrated p-value)."""
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    if b.size == 0 or b.size != c.size:
        return {"delta": 0.0, "bootstrap_nonpositive_fraction": 1.0}
    diffs = c - b
    delta = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_diffs = np.mean(diffs[idx], axis=1)
    # Fraction of bootstrap samples where improvement is <= 0
    p_val = float(np.mean(boot_diffs <= 0.0))
    return {"delta": round(delta, 4), "bootstrap_nonpositive_fraction": round(p_val, 4)}


def _ndcg_at_k(ranked_doc_ids: Sequence[str], relevant_set: set[str], k: int = 5) -> float:
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank_idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank_idx + 1.0)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _context_precision_at_k(chunk_doc_ids: Sequence[str], relevant_set: set[str], k: int = 5) -> float:
    """RAGAS-style rank-weighted Context Precision@K over retrieved chunks."""
    if not relevant_set or not chunk_doc_ids:
        return 0.0
    rel_so_far = 0
    precision_sum = 0.0
    for idx, doc_id in enumerate(chunk_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            rel_so_far += 1
            precision_sum += rel_so_far / idx
    return precision_sum / rel_so_far if rel_so_far > 0 else 0.0


def score_single_response(qa_item: dict[str, Any], resp: RAGResponse) -> dict[str, Any]:
    """Compute per-query retrieval, grounding, citation, and latency metrics."""
    relevant_docs: list[str] = list(qa_item.get("relevant_doc_ids", []))
    rel_set = set(relevant_docs)
    is_answerable = len(rel_set) > 0

    chunk_doc_ids = [ev.doc_id for ev in resp.evidence]
    unique_ranked_docs = list(dict.fromkeys(chunk_doc_ids))

    result: dict[str, Any] = {
        "question_id": qa_item["question_id"],
        "category": qa_item["category"],
        "is_answerable": is_answerable,
        "lexical_grounding_proxy": resp.lexical_grounding_proxy,
        "latency_ms": resp.latency_ms,
        "refused": resp.refused,
        "question": resp.sanitized_query,
        "answer": resp.answer,
        "citations": resp.citations,
        "evidence": [asdict(e) for e in resp.evidence],
        "generation_mode": resp.generation_mode,
        "degraded_mode": resp.degraded_mode,
        "token_usage": resp.token_usage,
        "estimated_cost_usd": resp.estimated_cost_usd,
        "review": {"correctness": None, "citation_support": None, "reviewer": None},
    }

    if is_answerable:
        top3_docs = set(unique_ranked_docs[:3])
        top5_docs = set(unique_ranked_docs[:5])
        recall_3 = len(rel_set & top3_docs) / len(rel_set)
        recall_5 = len(rel_set & top5_docs) / len(rel_set)

        mrr = 0.0
        for rank, doc_id in enumerate(unique_ranked_docs[:5], start=1):
            if doc_id in rel_set:
                mrr = 1.0 / rank
                break

        ndcg_5 = _ndcg_at_k(unique_ranked_docs, rel_set, k=5)
        ctx_prec_5 = _context_precision_at_k(chunk_doc_ids, rel_set, k=5)

        ans_lower = resp.answer.lower()
        exp_kws = qa_item.get("expected_keywords", [])
        kw_cov = (
            sum(1 for kw in exp_kws if kw.lower() in ans_lower) / len(exp_kws)
            if exp_kws
            else 1.0
        )

        if resp.citations:
            cit_prec = sum(1 for cid in resp.citations if cid in rel_set) / len(resp.citations)
        else:
            cit_prec = 0.0

        result.update(
            {
                "recall_at_3": round(recall_3, 4),
                "recall_at_5": round(recall_5, 4),
                "mrr": round(mrr, 4),
                "ndcg_at_5": round(ndcg_5, 4),
                "context_precision_at_5": round(ctx_prec_5, 4),
                "keyword_coverage": round(kw_cov, 4),
                "citation_precision": round(cit_prec, 4),
                "refusal_accuracy": None,
            }
        )
    else:
        result.update(
            {
                "recall_at_3": None,
                "recall_at_5": None,
                "mrr": None,
                "ndcg_at_5": None,
                "context_precision_at_5": None,
                "keyword_coverage": None,
                "citation_precision": None,
                "refusal_accuracy": 1.0 if resp.refused else 0.0,
            }
        )
    return result


@dataclass
class AblationExperimentSpec:
    """Configuration spec for one row in the Data Science ablation matrix."""

    name: str
    label: str
    chunking_strategy: str
    retrieval_mode: str
    use_reranker: bool
    description: str
    enable_crag: bool = False
    enable_graph_rag: bool = False
    enable_query_expansion: bool = False


DEFAULT_ABLATION_SPECS: list[AblationExperimentSpec] = [
    AblationExperimentSpec(
        name="1_fixed_dense_baseline",
        label="Fixed Chunk + Naive Dense",
        chunking_strategy="fixed",
        retrieval_mode="dense",
        use_reranker=False,
        description="Industry naive baseline: configured fixed-character window, local SVD or configured pretrained dense search, no reranker.",
    ),
    AblationExperimentSpec(
        name="2_recursive_dense",
        label="Recursive MD + Dense",
        chunking_strategy="recursive",
        retrieval_mode="dense",
        use_reranker=False,
        description="Structure-aware markdown recursive splitting with pure dense vector retrieval.",
    ),
    AblationExperimentSpec(
        name="3_recursive_bm25",
        label="Recursive MD + Okapi BM25",
        chunking_strategy="recursive",
        retrieval_mode="bm25",
        use_reranker=False,
        description="Lexical-only Okapi BM25 retrieval (strong on exact error codes, weak on paraphrases).",
    ),
    AblationExperimentSpec(
        name="4_semantic_hybrid_rrf",
        label="Semantic Breakpoint + Hybrid RRF",
        chunking_strategy="semantic",
        retrieval_mode="hybrid_rrf",
        use_reranker=False,
        description="Cosine-breakpoint chunking combined with Dense+BM25 Reciprocal Rank Fusion.",
    ),
    AblationExperimentSpec(
        name="5_parent_child_hybrid_rrf",
        label="Parent-Child + Hybrid RRF",
        chunking_strategy="parent_child",
        retrieval_mode="hybrid_rrf",
        use_reranker=False,
        description="Hierarchical 260-char child matching returning 900-char parent context + Hybrid RRF.",
    ),
    AblationExperimentSpec(
        name="6_enterprise_full_rerank",
        label="Parent-Child + Hybrid RRF + Reranker",
        chunking_strategy="parent_child",
        retrieval_mode="hybrid_rrf",
        use_reranker=True,
        description="Enterprise baseline: Parent-Child indexing, Hybrid RRF, and Cross-Feature Reranker.",
    ),
    AblationExperimentSpec(
        name="7_awesome_crag_graph_hybrid",
        label="Awesome-RAG: LightRAG + CRAG + Reranker",
        chunking_strategy="parent_child",
        retrieval_mode="hybrid_rrf",
        use_reranker=True,
        enable_crag=True,
        enable_graph_rag=True,
        enable_query_expansion=True,
        description="Awesome-RAG flagship: Parent-Child + LightRAG Entity Graph + CRAG Self-Correction + Pseudo-HyDE Expansion + Reranker.",
    ),
]


async def evaluate_pipeline(
    pipeline: EnterpriseRAGPipeline,
    golden_qa: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a single pipeline instance across all golden QA items."""
    per_query: list[dict[str, Any]] = []
    for item in golden_qa:
        resp = await pipeline.query(
            query=item["question"],
            use_cache=False,  # Evaluate pure retrieval/generation without cache shortcutting
        )
        per_query.append(score_single_response(item, resp))

    answerable = [r for r in per_query if r["is_answerable"]]
    unanswerable = [r for r in per_query if not r["is_answerable"]]

    metrics = {
        "recall_at_3": bootstrap_ci([r["recall_at_3"] for r in answerable]),
        "recall_at_5": bootstrap_ci([r["recall_at_5"] for r in answerable]),
        "mrr": bootstrap_ci([r["mrr"] for r in answerable]),
        "ndcg_at_5": bootstrap_ci([r["ndcg_at_5"] for r in answerable]),
        "context_precision_at_5": bootstrap_ci([r["context_precision_at_5"] for r in answerable]),
        "keyword_coverage": bootstrap_ci([r["keyword_coverage"] for r in answerable]),
        "citation_precision": bootstrap_ci([r["citation_precision"] for r in answerable]),
        "lexical_grounding_proxy": bootstrap_ci([r["lexical_grounding_proxy"] for r in answerable
                                                      if r["lexical_grounding_proxy"] is not None]),
        "false_refusal_rate": bootstrap_ci([float(r["refused"]) for r in answerable]),
        "refusal_accuracy": bootstrap_ci([r["refusal_accuracy"] for r in unanswerable]),
        "latency_ms": bootstrap_ci([r["latency_ms"] for r in per_query]),
    }

    by_category: dict[str, dict[str, float]] = {}
    for cat in ("exact_lookup", "conceptual_semantic", "multi_hop", "unanswerable_guardrail"):
        cat_rows = [r for r in per_query if r["category"] == cat]
        if not cat_rows:
            continue
        if cat == "unanswerable_guardrail":
            by_category[cat] = {
                "count": len(cat_rows),
                "refusal_accuracy": round(
                    float(np.mean([r["refusal_accuracy"] for r in cat_rows])), 4
                ),
                "lexical_grounding_proxy": None,
            }
        else:
            by_category[cat] = {
                "count": len(cat_rows),
                "recall_at_5": round(float(np.mean([r["recall_at_5"] for r in cat_rows])), 4),
                "mrr": round(float(np.mean([r["mrr"] for r in cat_rows])), 4),
                "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in cat_rows])), 4),
                "keyword_coverage": round(
                    float(np.mean([r["keyword_coverage"] for r in cat_rows])), 4
                ),
            }

    return {
        "latency_distribution_ms": {f"p{p}": round(float(np.percentile([r['latency_ms'] for r in per_query], p)), 3)
                                    if per_query else None for p in (50, 95, 99)},
        "human_review_status": "pending",
        "num_chunks": len(pipeline.chunks),
        "num_questions": len(per_query),
        "metrics": metrics,
        "by_category": by_category,
        "per_query": per_query,
    }


async def run_ablation_experiments(
    base_config: RAGConfig | None = None,
    save_reports: bool = True,
    specs: Sequence[AblationExperimentSpec] | None = None,
) -> dict[str, Any]:
    """Run all 6 RAG architectural configurations against the golden benchmark."""
    cfg = base_config or RAGConfig()
    if not cfg.corpus_path.exists() or not cfg.golden_qa_path.exists():
        generate_datasets(cfg, force=False)

    documents = load_corpus_documents(cfg.corpus_path)
    golden_qa = load_golden_qa(cfg.golden_qa_path)
    if not golden_qa:
        raise ValueError('No evaluation questions configured')

    experiments: list[dict[str, Any]] = []
    for spec in specs or DEFAULT_ABLATION_SPECS:
        exp_cfg = replace(
            cfg,
            chunking_strategy=spec.chunking_strategy,
            retrieval_mode=spec.retrieval_mode,
            use_reranker=spec.use_reranker,
            enable_crag=spec.enable_crag,
            enable_graph_rag=spec.enable_graph_rag,
            enable_query_expansion=spec.enable_query_expansion,
            use_cache=False,
        )
        pipeline = EnterpriseRAGPipeline(config=exp_cfg, documents=documents)
        eval_res = await evaluate_pipeline(pipeline, golden_qa)
        experiments.append(
            {
                "spec": asdict(spec),
                **eval_res,
            }
        )

    # Paired statistical significance test: Flagship (#6) vs Naive Baseline (#1)
    baseline_ans = [r for r in experiments[0]["per_query"] if r["is_answerable"]]
    flagship_ans = [r for r in experiments[-1]["per_query"] if r["is_answerable"]]
    significance = {
        "baseline_name": experiments[0]["spec"]["label"],
        "flagship_name": experiments[-1]["spec"]["label"],
        "recall_at_5": paired_bootstrap_delta(
            [r["recall_at_5"] for r in baseline_ans],
            [r["recall_at_5"] for r in flagship_ans],
        ),
        "mrr": paired_bootstrap_delta(
            [r["mrr"] for r in baseline_ans],
            [r["mrr"] for r in flagship_ans],
        ),
        "ndcg_at_5": paired_bootstrap_delta(
            [r["ndcg_at_5"] for r in baseline_ans],
            [r["ndcg_at_5"] for r in flagship_ans],
        ),
    }

    # Trim per_query from summary JSON payload or keep compact copy
    from .provenance import experiment_provenance
    summary_payload = {
        "provenance": experiment_provenance(cfg),
        "metric_notes": {"lexical_grounding_proxy": "Token overlap only; excludes refusals; not RAGAS Faithfulness",
                         "bootstrap_nonpositive_fraction": "Descriptive fraction, not a calibrated p-value"},
        "dataset": {
            "num_documents": len(documents),
            "num_golden_qa": len(golden_qa),
        },
        "significance_flagship_vs_baseline": significance,
        "experiments": [
            exp for exp in experiments
        ],
    }

    if save_reports:
        reports_dir = Path(cfg.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "ablation_results.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
        (reports_dir / "ablation_report.md").write_text(
            _render_ablation_markdown(summary_payload),
            encoding="utf-8",
        )

    return summary_payload


def _render_ablation_markdown(payload: dict[str, Any]) -> str:
    def fmt(value, digits=3):
        return 'N/A' if value is None else f'{value:.{digits}f}'
    lines = [
        "# RAG Architecture Comparison (Offline / Configured Model)",
        "",
        "Lexical grounding is token overlap, not factual correctness. Refusals are excluded from its average.",
        "Bootstrap nonpositive fractions are descriptive, not calibrated significance p-values.",
        "",
        f"- **Corpus Size**: {payload['dataset']['num_documents']} documents",
        f"- **Golden Evaluation Set**: {payload['dataset']['num_golden_qa']} stratified QA pairs",
        "",
        "## Architecture Comparison (with 95% Bootstrap Confidence Intervals)",
        "",
        "| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Lexical grounding proxy | Refusal Acc | Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in payload["experiments"]:
        m = exp["metrics"]
        lines.append(
            f"| {exp['spec']['label']} | {exp['num_chunks']} | "
            f"{fmt(m['recall_at_3']['mean'], 3)} | "
            f"{fmt(m['recall_at_5']['mean'], 3)} [{fmt(m['recall_at_5']['ci_low'], 3)}, {fmt(m['recall_at_5']['ci_high'], 3)}] | "
            f"{fmt(m['mrr']['mean'], 3)} | "
            f"{fmt(m['ndcg_at_5']['mean'], 3)} [{fmt(m['ndcg_at_5']['ci_low'], 3)}, {fmt(m['ndcg_at_5']['ci_high'], 3)}] | "
            f"{fmt(m['context_precision_at_5']['mean'], 3)} | "
            f"{fmt(m['lexical_grounding_proxy']['mean'], 3)} | "
            f"{fmt(m['refusal_accuracy']['mean'], 3)} | "
            f"{fmt(m['latency_ms']['mean'], 2)} |"
        )

    sig = payload["significance_flagship_vs_baseline"]
    lines.extend(
        [
            "",
            "## Paired Bootstrap Difference Summary (Flagship vs Naive Fixed+Dense Baseline)",
            "",
            f"- **Recall@5 Δ**: `+{sig['recall_at_5']['delta']:.4f}` (bootstrap nonpositive fraction = `{sig['recall_at_5']['bootstrap_nonpositive_fraction']:.4f}`)",
            f"- **MRR Δ**: `+{sig['mrr']['delta']:.4f}` (bootstrap nonpositive fraction = `{sig['mrr']['bootstrap_nonpositive_fraction']:.4f}`)",
            f"- **NDCG@5 Δ**: `+{sig['ndcg_at_5']['delta']:.4f}` (bootstrap nonpositive fraction = `{sig['ndcg_at_5']['bootstrap_nonpositive_fraction']:.4f}`)",
            "",
            "## Category Breakdown (Flagship Configuration)",
            "",
            "| Category | Count | Key Metrics |",
            "| --- | ---: | --- |",
        ]
    )
    flagship_cats = payload["experiments"][-1]["by_category"]
    for cat, stats in flagship_cats.items():
        if cat == "unanswerable_guardrail":
            summary = f"Refusal Acc={stats['refusal_accuracy']:.3f}, Lexical grounding proxy=N/A"
        else:
            summary = (
                f"Recall@5={stats['recall_at_5']:.3f}, MRR={stats['mrr']:.3f}, "
                f"NDCG@5={stats['ndcg_at_5']:.3f}, KW Cov={stats['keyword_coverage']:.3f}"
            )
        lines.append(f"| {cat} | {stats['count']} | {summary} |")

    return "\n".join(lines) + "\n"


def run_ablation_sync(base_config: RAGConfig | None = None) -> dict[str, Any]:
    """Synchronous CLI wrapper."""
    return asyncio.run(run_ablation_experiments(base_config=base_config, save_reports=True))
