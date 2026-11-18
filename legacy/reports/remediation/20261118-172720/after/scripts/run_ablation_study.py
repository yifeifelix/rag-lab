"""CLI entrypoint for running the 6-architecture Data Science ablation study."""

from __future__ import annotations

from rag_platform.evaluation import run_ablation_sync


def main() -> None:
    print("Running 7-architecture RAG ablation study on the configured corpus and QA set...")
    result = run_ablation_sync()
    print("\n=== Ablation Summary ===")
    for exp in result["experiments"]:
        m = exp["metrics"]
        print(
            f"{exp['spec']['label']:40s} | "
            f"R@5={m['recall_at_5']['mean']:.3f} | "
            f"MRR={m['mrr']['mean']:.3f} | "
            f"NDCG@5={m['ndcg_at_5']['mean']:.3f} | "
            f"LexicalProxy={m['lexical_grounding_proxy']['mean']:.3f} | "
            f"Refusal={m['refusal_accuracy']['mean']:.3f}"
        )
    sig = result["significance_flagship_vs_baseline"]
    print(
        f"\nFlagship vs Naive Baseline: "
        f"Recall@5 +{sig['recall_at_5']['delta']:.4f} (bootstrap fraction={sig['recall_at_5']['bootstrap_nonpositive_fraction']:.4f}), "
        f"MRR +{sig['mrr']['delta']:.4f} (bootstrap fraction={sig['mrr']['bootstrap_nonpositive_fraction']:.4f}), "
        f"NDCG@5 +{sig['ndcg_at_5']['delta']:.4f} (bootstrap fraction={sig['ndcg_at_5']['bootstrap_nonpositive_fraction']:.4f})"
    )


if __name__ == "__main__":
    main()
