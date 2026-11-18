FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small (no CUDA).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install \
    "sentence-transformers==3.3.1" "transformers==4.46.3" \
    "jupyterlab==4.3.4" "jupytext==1.16.6" "ipywidgets==8.1.5" \
    "jupyter-book==1.0.3" \
    "numpy==1.26.4" "pandas==2.2.3" "matplotlib==3.9.3" "scikit-learn==1.5.2" \
    "openai==1.57.4" "pytest==8.3.4"

# Bake the two small models into the image so lessons work offline.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
PY
RUN chmod -R a+rX /opt/models

# Open .md lessons as notebooks by default.
RUN mkdir -p /etc/jupyter/labconfig && printf '%s' \
    '{"@jupyterlab/docmanager-extension:plugin":{"defaultViewers":{"markdown":"Jupytext Notebook","myst":"Jupytext Notebook"}}}' \
    > /etc/jupyter/labconfig/default_setting_overrides.json

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/home/lab/project/src \
    RAGLAB_CACHE=/home/lab/project/.cache/raglab \
    HOME=/home/lab/project/.cache/home
WORKDIR /home/lab/project
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "mkdir -p \"$HOME\" && exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/home/lab/project --ServerApp.default_url=/lab/tree/workspace/00_welcome.md"]
