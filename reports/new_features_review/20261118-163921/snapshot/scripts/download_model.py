"""Explicit model download, kept in the workbench's Data-volume cache."""

import argparse
from rag_platform.config import RAGConfig
from rag_platform.embeddings import PretrainedEmbedder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=RAGConfig().embedding_model)
args = parser.parse_args()
encoder = PretrainedEmbedder(args.model, allow_download=True)
print('Downloaded/loaded model:', args.model)
print('Smoke embedding shape:', encoder.encode(['A local model cache verification.']).shape)
