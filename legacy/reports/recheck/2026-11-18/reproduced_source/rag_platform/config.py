"""Configuration dataclasses for the Enterprise RAG Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from .runtime import configure_runtime

configure_runtime()


def get_project_root() -> Path:
    """Return the root directory of the RAG repository."""
    return Path(os.getenv('RAG_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))


@dataclass
class RAGConfig:
    """Central configuration for chunking, retrieval, caching, resilience, and generation."""

    # Paths (kept strictly under project root on /Volumes/Data)
    project_root: Path = field(default_factory=get_project_root)
    corpus_path: Path | None = None
    golden_qa_path: Path | None = None
    reports_dir: Path | None = None
    index_dir: Path | None = None
    persist_index: bool = True
    bootstrap_synthetic: bool = field(default_factory=lambda: os.getenv('RAG_BOOTSTRAP_SYNTHETIC', 'true').lower() == 'true')
    admin_token: str = field(default_factory=lambda: os.getenv('RAG_ADMIN_TOKEN', ''))

    # Chunking parameters
    chunking_strategy: str = "parent_child"  # fixed | recursive | semantic | parent_child
    fixed_chunk_size: int = 260
    fixed_overlap: int = 40
    recursive_max_chars: int = 500
    semantic_breakpoint_percentile: float = 80.0
    parent_max_chars: int = 900
    child_max_chars: int = 260

    # Embedding & retrieval parameters
    embedding_dim: int = 48
    embedding_backend: str = field(default_factory=lambda: os.getenv('RAG_EMBEDDING_BACKEND', 'local_svd'))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        'RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    allow_model_download: bool = False
    retrieval_mode: str = "hybrid_rrf"  # dense | bm25 | hybrid_rrf
    top_k_candidates: int = 36
    top_k_final: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.55
    sparse_weight: float = 0.45

    # Reranker & relevance threshold
    use_reranker: bool = True
    min_confidence_threshold: float = 0.22

    # Multi-tier caching & coalescing
    use_cache: bool = True
    l1_max_size: int = 4096
    l1_ttl_seconds: float = 3600.0
    l2_max_size: int = 2048
    l2_similarity_threshold: float = 0.90
    l2_ttl_seconds: float = 3600.0
    enable_singleflight: bool = True

    # 10k-Concurrency resilience & backpressure
    max_concurrent_pipelines: int = 256
    queue_timeout_seconds: float = 5.0
    rate_limit_rps: float = 50000.0
    rate_limit_burst: float = 20000.0
    circuit_breaker_failure_threshold: int = 10
    circuit_breaker_reset_seconds: float = 15.0

    # Synthesis / LLM mode
    llm_mode: str = field(default_factory=lambda: os.getenv("RAG_LLM_MODE", "local"))
    simulated_llm_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.corpus_path is None:
            self.corpus_path = Path(os.getenv('RAG_CORPUS_PATH', str(self.project_root / 'data/workspaces/demo/documents.jsonl')))
        if self.golden_qa_path is None:
            self.golden_qa_path = Path(os.getenv('RAG_QA_PATH', str(self.project_root / 'data/eval/golden_qa.jsonl')))
        if self.reports_dir is None:
            self.reports_dir = Path(os.getenv('RAG_REPORTS_DIR', str(self.project_root / 'reports')))
        if self.index_dir is None:
            self.index_dir = Path(os.getenv('RAG_INDEX_DIR', str(self.corpus_path.parent / 'index')))
        if self.embedding_backend not in ('local_svd', 'sentence_transformers'):
            raise ValueError('Unsupported embedding backend')
        if self.llm_mode not in ('local', 'openai', 'gemini'):
            raise ValueError('Unsupported LLM mode')
