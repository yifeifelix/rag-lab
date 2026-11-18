"""CLI entrypoint for running the 6-architecture Data Science ablation study."""

from __future__ import annotations

from rag_platform.evaluation import run_ablation_sync


def main() -> None:
    print("Running 6-architecture Enterprise RAG ablation study across 40 golden QA pairs...")
    result = run_ablation_sync()
    print("\n=== Ablation Summary ===")
    for exp in result["experiments"]:
        m = exp["metrics"]
        print(
            f"{exp['spec']['label']:40s} | "
            f"R@5={m['recall_at_5']['mean']:.3f} | "
            f"MRR={m['mrr']['mean']:.3f} | "
            f"NDCG@5={m['ndcg_at_5']['mean']:.3f} | "
            f"Faith={m['faithfulness']['mean']:.3f} | "
            f"Refusal={m['refusal_accuracy']['mean']:.3f}"
        )
    sig = result["significance_flagship_vs_baseline"]
    print(
        f"\nFlagship vs Naive Baseline: "
        f"Recall@5 +{sig['recall_at_5']['delta']:.4f} (p={sig['recall_at_5']['p_value']:.4f}), "
        f"MRR +{sig['mrr']['delta']:.4f} (p={sig['mrr']['p_value']:.4f}), "
        f"NDCG@5 +{sig['ndcg_at_5']['delta']:.4f} (p={sig['ndcg_at_5']['p_value']:.4f})"
    )


if __name__ == "__main__":
    main()
