"""Side-effect-free configuration definitions for the local RAG workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


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
    dev_qa_path: Path | None = None
    test_qa_path: Path | None = None
    calibration_path: Path | None = None
    use_calibration: bool = True
    reports_dir: Path | None = None
    index_dir: Path | None = None
    persist_index: bool = True
    bootstrap_synthetic: bool = field(default_factory=lambda: os.getenv('RAG_BOOTSTRAP_SYNTHETIC', 'true').lower() == 'true')
    admin_token: str = field(default_factory=lambda: os.getenv('RAG_ADMIN_TOKEN', ''))
    tester_token: str = field(default_factory=lambda: os.getenv('RAG_TESTER_TOKEN', ''))
    cpu_workers: int = field(default_factory=lambda: int(os.getenv('RAG_CPU_WORKERS', '4')))

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
    embedding_revision: str = field(default_factory=lambda: os.getenv('RAG_EMBEDDING_REVISION', ''))
    vector_backend: str = field(default_factory=lambda: os.getenv('RAG_VECTOR_BACKEND', 'memory'))
    qdrant_url: str = field(default_factory=lambda: os.getenv('RAG_QDRANT_URL', 'http://qdrant:6333'))
    qdrant_collection: str = field(default_factory=lambda: os.getenv('RAG_QDRANT_COLLECTION', 'rag_workbench'))
    allow_model_download: bool = False
    retrieval_mode: str = "hybrid_rrf"  # dense | bm25 | hybrid_rrf
    top_k_candidates: int = 36
    top_k_final: int = 5
    rrf_k: int = field(default_factory=lambda: int(os.getenv('RAG_RRF_K', '60')))
    dense_weight: float = field(default_factory=lambda: float(os.getenv('RAG_DENSE_WEIGHT', '0.55')))
    sparse_weight: float = field(default_factory=lambda: float(os.getenv('RAG_SPARSE_WEIGHT', '0.45')))
    graph_weight: float = field(default_factory=lambda: float(os.getenv('RAG_GRAPH_WEIGHT', '0.20')))
    bm25_scale: float = 18.0
    query_vocab_floor: float = 0.45
    grounding_min_terms: int = 3
    dense_confidence_weights: tuple[float, float] = (0.60, 0.40)
    sparse_confidence_weights: tuple[float, float] = (0.65, 0.35)
    hybrid_confidence_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)

    # Reranker & relevance threshold
    use_reranker: bool = True
    min_confidence_threshold: float = field(default_factory=lambda: float(os.getenv('RAG_CONFIDENCE_THRESHOLD', '0.22')))
    reranker_backend: str = field(default_factory=lambda: os.getenv('RAG_RERANKER_BACKEND', 'features'))
    reranker_model: str = field(default_factory=lambda: os.getenv('RAG_RERANKER_MODEL', 'BAAI/bge-reranker-base'))
    reranker_revision: str = field(default_factory=lambda: os.getenv('RAG_RERANKER_REVISION', ''))
    feature_rank_weights: tuple[float, ...] = (0.24, 0.16, 0.16, 0.22, 0.10, 0.06, 0.06)
    feature_confidence_weights: tuple[float, ...] = (0.35, 0.25, 0.25, 0.15)

    # Awesome-RAG techniques (opt-in flags; tested in Ablation #7 & AutoRAG tuner)
    enable_crag: bool = field(default_factory=lambda: os.getenv('RAG_ENABLE_ROUTER', 'false').lower() == 'true')
    enable_graph_rag: bool = field(default_factory=lambda: os.getenv('RAG_ENABLE_GRAPH', 'false').lower() == 'true')
    enable_query_expansion: bool = field(default_factory=lambda: os.getenv('RAG_ENABLE_EXPANSION', 'false').lower() == 'true')
    enable_contextual_compression: bool = field(default_factory=lambda: os.getenv('RAG_ENABLE_COMPRESSION', 'false').lower() == 'true')
    enable_contextual_headers: bool = field(default_factory=lambda: os.getenv('RAG_ENABLE_HEADERS', 'false').lower() == 'true')
    expansion_rules_path: Path | None = None
    crag_ambiguous_threshold: float = 0.48
    crag_vocab_threshold: float = 0.35

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
        self.project_root = Path(self.project_root).expanduser().resolve()
        if self.corpus_path is None:
            self.corpus_path = Path(os.getenv('RAG_CORPUS_PATH', str(self.project_root / 'data/workspaces/user_testing_v2/documents.jsonl')))
        if self.golden_qa_path is None:
            default_qa = self.project_root / 'data/learning_v2/qa_dev.jsonl'
            if not default_qa.exists():
                default_qa = self.project_root / 'data/eval/golden_qa.jsonl'
            self.golden_qa_path = Path(os.getenv('RAG_QA_PATH', str(default_qa)))
        if self.dev_qa_path is None:
            self.dev_qa_path = Path(os.getenv('RAG_DEV_QA_PATH', str(self.project_root / 'data/learning_v2/qa_dev.jsonl')))
        if self.test_qa_path is None:
            self.test_qa_path = Path(os.getenv('RAG_TEST_QA_PATH', str(self.project_root / 'data/learning_v2/qa_test.jsonl')))
        if self.calibration_path is None and os.getenv('RAG_CALIBRATION_PATH'):
            self.calibration_path = Path(os.environ['RAG_CALIBRATION_PATH'])
        if self.reports_dir is None:
            self.reports_dir = Path(os.getenv('RAG_REPORTS_DIR', str(self.project_root / 'reports/private')))
        if self.index_dir is None:
            self.index_dir = Path(os.getenv('RAG_INDEX_DIR', str(self.corpus_path.parent / 'index')))
        if self.embedding_backend not in ('local_svd', 'sentence_transformers'):
            raise ValueError('Unsupported embedding backend')
        if self.llm_mode not in ('local', 'openai', 'gemini'):
            raise ValueError('Unsupported LLM mode')
        if self.reranker_backend not in ('features', 'cross_encoder') or self.vector_backend not in ('memory', 'qdrant'):
            raise ValueError('Unsupported reranker/vector backend')
        if self.vector_backend == 'qdrant' and self.embedding_backend != 'sentence_transformers':
            raise ValueError('Qdrant incremental updates require a fixed pretrained encoder')
        if self.cpu_workers < 1 or not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError('Invalid execution/threshold settings')
        if min(self.dense_weight, self.sparse_weight, self.graph_weight) < 0:
            raise ValueError('RRF weights must be non-negative')
        data_root = Path(os.getenv('RAG_DATA_ROOT', '/Volumes/Data')).expanduser().resolve()
        for key in ('project_root', 'corpus_path', 'golden_qa_path', 'dev_qa_path', 'test_qa_path',
                    'reports_dir', 'index_dir', 'calibration_path', 'expansion_rules_path'):
            value = getattr(self, key)
            if value is not None:
                value = Path(value).expanduser().resolve()
                if not value.is_relative_to(data_root):
                    raise ValueError(f'{key} must reside under RAG_DATA_ROOT ({data_root})')
                setattr(self, key, value)
