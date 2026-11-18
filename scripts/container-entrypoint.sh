#!/bin/sh
set -eu
mkdir -p "$RAG_REPORTS_DIR"
if [ "${RAG_EMBEDDING_BACKEND:-local_svd}" = sentence_transformers ]; then
  manifest=/app/.cache/model_versions.json
  RAG_EMBEDDING_REVISION=$(python -c 'import json,re,sys; m=json.load(open(sys.argv[1])); assert m["embedding_model"]==sys.argv[2]; r=m["embedding_revision"]; assert re.fullmatch("[a-f0-9]{40,64}",r); print(r)' "$manifest" "$RAG_EMBEDDING_MODEL")
  RAG_RERANKER_REVISION=$(python -c 'import json,re,sys; m=json.load(open(sys.argv[1])); assert m["reranker_model"]==sys.argv[2]; r=m["reranker_revision"]; assert re.fullmatch("[a-f0-9]{40,64}",r); print(r)' "$manifest" "$RAG_RERANKER_MODEL")
  export RAG_EMBEDDING_REVISION RAG_RERANKER_REVISION
fi
if [ "${RAG_VECTOR_BACKEND:-memory}" = qdrant ]; then
  python - <<'PY'
import os,time,urllib.request
for attempt in range(60):
    try:
        urllib.request.urlopen(os.environ['RAG_QDRANT_URL'] + '/readyz', timeout=2).close()
        break
    except OSError:
        if attempt==59: raise
        time.sleep(1)
PY
fi
if [ -n "${RAG_CALIBRATION_PATH:-}" ]; then
  if ! RAG_CALIBRATION_PATH= python /app/scripts/run_learning_study.py calibrate --output "$RAG_CALIBRATION_PATH"; then
    echo 'Development calibration unavailable; the UI will label the score as uncalibrated.' >&2
  fi
fi
exec "$@"
