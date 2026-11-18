#!/bin/sh
# Shallow-clone the community RAG learning repositories into references/.
# They are for private study only: several have no open-source licence, so the
# folder is git-ignored and never published with this project.
set -eu
cd "$(dirname "$0")/../references"
while read -r repo; do
  [ -z "$repo" ] && continue
  dir=$(echo "$repo" | tr '/' '__')
  if [ -d "$dir/.git" ]; then
    echo "update $repo"; git -C "$dir" pull --ff-only --depth 1 -q || true
  else
    echo "clone  $repo"; git clone --depth 1 -q "https://github.com/$repo.git" "$dir" || echo "  failed: $repo"
  fi
done <<'EOF'
langchain-ai/rag-from-scratch
NirDiamant/RAG_Techniques
mrdbourke/simple-local-rag
KalyanKS-NLP/rag-zero-to-hero-guide
datawhalechina/all-in-rag
GiovanniPasq/agentic-rag-for-dummies
pguso/rag-from-scratch
IlyaRice/RAG-Challenge-2
HKUDS/LightRAG
VectifyAI/PageIndex
EOF
