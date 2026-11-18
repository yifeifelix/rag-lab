"""Distributed Locust load-testing script for HTTP benchmarking against FastAPI pods.

Usage:
    locust -f deploy/locustfile.py --host http://localhost:8000 -u 10000 -r 500
"""

from __future__ import annotations

import random
from locust import HttpUser, between, task

HOT_QUERIES = [
    "What does error code ERR_QDRANT_5032 indicate and how is it recovered?",
    "Which security controls govern SOC2-CC6.7 and ISO27001-A.8.24 in the vector layer?",
    "What does PCI-DSS-3.5.1 prohibit from entering enterprise RAG knowledge bases?",
    "What are the statutory floor and early-warning thresholds under FIN-BASEL-LCR-105?",
    "What are the uptime and latency numbers in SLA-TIER0-99.99?",
]

PARAPHRASE_QUERIES = [
    "Explain the vector layer requirements under SOC2-CC6.7 and ISO27001-A.8.24.",
    "What payment data is banned from RAG ingestion under PCI-DSS-3.5.1?",
    "List the Basel III LCR regulatory floor and internal triggers in FIN-BASEL-LCR-105.",
    "Why is vLLM continuous batching faster than static request batching?",
    "How does hierarchical chunking feed into Hybrid RRF and reranking?",
]


class EnterpriseRAGUser(HttpUser):
    """Simulates Zipf enterprise traffic against /api/v1/query."""

    wait_time = between(0.2, 1.0)

    @task(45)
    def hot_exact_query(self) -> None:
        q = random.choice(HOT_QUERIES)
        self.client.post(
            "/api/v1/query",
            json={"query": q, "retrieval_mode": "hybrid_rrf", "use_reranker": True, "use_cache": True},
            name="/api/v1/query [hot-L1]",
        )

    @task(30)
    def paraphrase_semantic_query(self) -> None:
        q = random.choice(PARAPHRASE_QUERIES)
        self.client.post(
            "/api/v1/query",
            json={"query": q, "retrieval_mode": "hybrid_rrf", "use_reranker": True, "use_cache": True},
            name="/api/v1/query [semantic-L2]",
        )

    @task(25)
    def cold_tail_query(self) -> None:
        nonce = random.randint(10000, 99999)
        self.client.post(
            "/api/v1/query",
            json={
                "query": f"Audit request #{nonce}: summarize Little's Law capacity and circuit breaker SLA",
                "retrieval_mode": "hybrid_rrf",
                "use_reranker": True,
                "use_cache": False,
            },
            name="/api/v1/query [cold-tail]",
        )
