"""Data Science RAG evaluation framework: retrieval, faithfulness, bootstrap CIs & ablation runner."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
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
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return {"mean": round(mean_val, 4), "ci_low": round(mean_val, 4), "ci_high": round(mean_val, 4)}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return {
        "mean": round(mean_val, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
    }


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute paired mean improvement and one-sided bootstrap p-value (H1: candidate > baseline)."""
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    if b.size == 0 or b.size != c.size:
        return {"delta": 0.0, "p_value": 1.0}
    diffs = c - b
    delta = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_diffs = np.mean(diffs[idx], axis=1)
    # Fraction of bootstrap samples where improvement is <= 0
    p_val = float(np.mean(boot_diffs <= 0.0))
    return {"delta": round(delta, 4), "p_value": round(p_val, 4)}


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
        "faithfulness": resp.faithfulness,
        "latency_ms": resp.latency_ms,
        "refused": resp.refused,
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


DEFAULT_ABLATION_SPECS: list[AblationExperimentSpec] = [
    AblationExperimentSpec(
        name="1_fixed_dense_baseline",
        label="Fixed Chunk + Naive Dense",
        chunking_strategy="fixed",
        retrieval_mode="dense",
        use_reranker=False,
        description="Industry naive baseline: 400-char fixed window, pure bi-encoder dense search, no reranker.",
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
        description="Flagship Enterprise RAG: Parent-Child indexing, Hybrid RRF, and Cross-Feature Reranker.",
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
        "faithfulness": bootstrap_ci([r["faithfulness"] for r in per_query]),
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
                "faithfulness": round(float(np.mean([r["faithfulness"] for r in cat_rows])), 4),
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
        "num_chunks": len(pipeline.chunks),
        "num_questions": len(per_query),
        "metrics": metrics,
        "by_category": by_category,
        "per_query": per_query,
    }


async def run_ablation_experiments(
    base_config: RAGConfig | None = None,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Run all 6 RAG architectural configurations against the golden benchmark."""
    cfg = base_config or RAGConfig()
    if not cfg.corpus_path.exists() or not cfg.golden_qa_path.exists():
        generate_datasets(cfg, force=False)

    documents = load_corpus_documents(cfg.corpus_path)
    golden_qa = load_golden_qa(cfg.golden_qa_path)

    experiments: list[dict[str, Any]] = []
    for spec in DEFAULT_ABLATION_SPECS:
        exp_cfg = RAGConfig(
            project_root=cfg.project_root,
            corpus_path=cfg.corpus_path,
            golden_qa_path=cfg.golden_qa_path,
            reports_dir=cfg.reports_dir,
            chunking_strategy=spec.chunking_strategy,
            retrieval_mode=spec.retrieval_mode,
            use_reranker=spec.use_reranker,
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
    summary_payload = {
        "dataset": {
            "num_documents": len(documents),
            "num_golden_qa": len(golden_qa),
        },
        "significance_flagship_vs_baseline": significance,
        "experiments": [
            {k: v for k, v in exp.items() if k != "per_query"} for exp in experiments
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
    lines = [
        "# Enterprise RAG Data Science Ablation Report",
        "",
        f"- **Corpus Size**: {payload['dataset']['num_documents']} enterprise documents",
        f"- **Golden Evaluation Set**: {payload['dataset']['num_golden_qa']} stratified QA pairs",
        "",
        "## Architecture Comparison (with 95% Bootstrap Confidence Intervals)",
        "",
        "| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Faithfulness | Refusal Acc | Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in payload["experiments"]:
        m = exp["metrics"]
        lines.append(
            f"| {exp['spec']['label']} | {exp['num_chunks']} | "
            f"{m['recall_at_3']['mean']:.3f} | "
            f"{m['recall_at_5']['mean']:.3f} [{m['recall_at_5']['ci_low']:.3f}, {m['recall_at_5']['ci_high']:.3f}] | "
            f"{m['mrr']['mean']:.3f} | "
            f"{m['ndcg_at_5']['mean']:.3f} [{m['ndcg_at_5']['ci_low']:.3f}, {m['ndcg_at_5']['ci_high']:.3f}] | "
            f"{m['context_precision_at_5']['mean']:.3f} | "
            f"{m['faithfulness']['mean']:.3f} | "
            f"{m['refusal_accuracy']['mean']:.3f} | "
            f"{m['latency_ms']['mean']:.2f} |"
        )

    sig = payload["significance_flagship_vs_baseline"]
    lines.extend(
        [
            "",
            "## Paired Significance Test (Flagship vs Naive Fixed+Dense Baseline)",
            "",
            f"- **Recall@5 Δ**: `+{sig['recall_at_5']['delta']:.4f}` (p-value = `{sig['recall_at_5']['p_value']:.4f}`)",
            f"- **MRR Δ**: `+{sig['mrr']['delta']:.4f}` (p-value = `{sig['mrr']['p_value']:.4f}`)",
            f"- **NDCG@5 Δ**: `+{sig['ndcg_at_5']['delta']:.4f}` (p-value = `{sig['ndcg_at_5']['p_value']:.4f}`)",
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
            summary = f"Refusal Acc={stats['refusal_accuracy']:.3f}, Faithfulness={stats['faithfulness']:.3f}"
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
