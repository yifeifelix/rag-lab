FROM python:3.12-slim-bookworm AS builder
WORKDIR /build
COPY requirements/build.lock /build/build.lock
RUN pip install --no-cache-dir --require-hashes -r build.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV RAG_DATA_ROOT=/app RAG_PROJECT_ROOT=/app \
    RAG_LLM_MODE=local RAG_EMBEDDING_BACKEND=local_svd \
    RAG_REPORTS_DIR=/app/reports/private \
    RAG_CALIBRATION_PATH="" \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/app
COPY requirements/runtime.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels \
    && groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --no-create-home rag
COPY data/corpus/documents.jsonl /app/data/corpus/documents.jsonl
COPY data/eval/golden_qa.jsonl /app/data/eval/golden_qa.jsonl
COPY data/learning_v2 /app/data/learning_v2
COPY data/python_support/PYTHON_LICENSE.txt /app/data/python_support/PYTHON_LICENSE.txt
COPY THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md
COPY scripts/run_learning_study.py scripts/download_model.py scripts/integration_qdrant.py /app/scripts/
COPY scripts/container-entrypoint.sh /usr/local/bin/rag-entrypoint
RUN mkdir -p /app/data/workspaces /app/reports/private /app/.cache \
    && chown -R rag:rag /app && chmod 755 /usr/local/bin/rag-entrypoint
USER rag

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=10s --start-period=90s --retries=6 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=8)); assert r['status']=='healthy' and r['indexed_chunks']>0"
ENTRYPOINT ["rag-entrypoint"]
CMD ["python", "-m", "uvicorn", "rag_platform.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM runtime AS models
USER root
# This optional lock is resolved for the host's Linux architecture by scripts/lock-models.
COPY requirements/models.lock /app/models.lock
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu --require-hashes -r /app/models.lock
USER rag
# A plain docker build uses the offline runtime; models remain an explicit target.
FROM runtime AS default
