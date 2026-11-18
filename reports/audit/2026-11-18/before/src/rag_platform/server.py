"""FastAPI Enterprise RAG Server with SSE streaming, Ablation API & 10k Benchmark runner."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .benchmark import run_concurrency_benchmark
from .config import RAGConfig
from .data_gen import generate_datasets, load_golden_qa
from .evaluation import run_ablation_experiments
from .pipeline import EnterpriseRAGPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    retrieval_mode: str = Field(default="hybrid_rrf", pattern="^(dense|bm25|hybrid_rrf)$")
    use_reranker: bool = True
    use_cache: bool = True
    top_k: int = Field(default=5, ge=1, le=15)
    domain_filter: str | None = None


class BenchmarkRequest(BaseModel):
    total_requests: int = Field(default=2000, ge=100, le=10000)
    simulated_llm_latency_ms: float = Field(default=2.0, ge=0.0, le=50.0)


def create_app(config: RAGConfig | None = None) -> FastAPI:
    """Create and configure the Enterprise RAG FastAPI application."""
    cfg = config or RAGConfig()
    if not cfg.corpus_path.exists() or not cfg.golden_qa_path.exists():
        generate_datasets(cfg, force=False)

    pipeline = EnterpriseRAGPipeline(config=cfg)
    golden_qa = load_golden_qa(cfg.golden_qa_path)

    app = FastAPI(
        title="Enterprise 10k-Concurrency RAG Platform & DS Workbench",
        version="1.0.0",
        description="Hybrid RRF + Cross-Feature Reranker + L1/L2 Semantic Cache + Singleflight + Evaluation Workbench",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = cfg
    app.state.pipeline = pipeline
    app.state.golden_qa = golden_qa

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Workbench UI not found")
        return FileResponse(index_path)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "llm_mode": cfg.llm_mode,
            "corpus_documents": len(pipeline.documents),
            "indexed_chunks": len(pipeline.chunks),
            "chunking_strategy": cfg.chunking_strategy,
        }

    @app.post("/api/v1/query")
    async def query_rag(req: QueryRequest) -> dict[str, Any]:
        resp = await pipeline.query(
            query=req.query,
            retrieval_mode=req.retrieval_mode,
            use_reranker=req.use_reranker,
            use_cache=req.use_cache,
            top_k=req.top_k,
            domain_filter=req.domain_filter,
        )
        return asdict(resp)

    @app.post("/api/v1/query/stream")
    async def stream_rag(req: QueryRequest) -> StreamingResponse:
        async def _event_generator():
            async for evt in pipeline.stream_query(
                query=req.query,
                retrieval_mode=req.retrieval_mode,
                use_reranker=req.use_reranker,
                use_cache=req.use_cache,
                top_k=req.top_k,
                domain_filter=req.domain_filter,
            ):
                yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    @app.get("/api/v1/ablation")
    async def get_ablation(refresh: bool = False) -> dict[str, Any]:
        report_file = Path(cfg.reports_dir) / "ablation_results.json"
        if report_file.exists() and not refresh:
            return json.loads(report_file.read_text(encoding="utf-8"))
        return await run_ablation_experiments(base_config=cfg, save_reports=True)

    @app.get("/api/v1/benchmark")
    async def get_benchmark() -> dict[str, Any]:
        bench_file = Path(cfg.reports_dir) / "benchmark_10k.json"
        if bench_file.exists():
            return json.loads(bench_file.read_text(encoding="utf-8"))
        return await run_concurrency_benchmark(total_requests=10000, base_config=cfg, save_reports=True)

    @app.post("/api/v1/benchmark/run")
    async def run_live_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
        save = req.total_requests == 10000
        return await run_concurrency_benchmark(
            total_requests=req.total_requests,
            simulated_llm_latency_ms=req.simulated_llm_latency_ms,
            save_reports=save,
        )

    @app.get("/api/v1/stats")
    async def get_stats() -> dict[str, Any]:
        samples = [
            {
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["question"],
                "paraphrase": q["paraphrase_variants"][0] if q.get("paraphrase_variants") else q["question"],
            }
            for q in golden_qa[:16]
        ] + [
            {
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["question"],
                "paraphrase": q["paraphrase_variants"][0] if q.get("paraphrase_variants") else q["question"],
            }
            for q in golden_qa
            if q["category"] in ("multi_hop", "unanswerable_guardrail")
        ][:8]
        return {
            "telemetry": pipeline.get_stats(),
            "domains": sorted({d["domain"] for d in pipeline.documents}),
            "sample_queries": samples,
        }

    return app


app = create_app()
