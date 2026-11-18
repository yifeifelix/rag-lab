"""Explicit, version-recorded downloads and actual encoder/reranker smoke inference."""
import argparse
import json
from pathlib import Path
import re

from rag_platform.config import RAGConfig
from rag_platform.embeddings import PretrainedEmbedder
from rag_platform.runtime import configure_runtime


def main():
    configure_runtime()
    cfg = RAGConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=cfg.embedding_model)
    parser.add_argument('--revision', default=cfg.embedding_revision)
    parser.add_argument('--reranker', default=cfg.reranker_model)
    parser.add_argument('--reranker-revision', default=cfg.reranker_revision)
    parser.add_argument('--both', action='store_true')
    args = parser.parse_args()
    encoder = PretrainedEmbedder(args.model, allow_download=True, revision=args.revision)
    shape = list(encoder.encode(['A verified local model cache.']).shape)
    manifest = {'embedding_model': args.model, 'embedding_revision': encoder.revision,
                'embedding_smoke_shape': shape, 'real_embedding_inference': True}
    if args.both:
        from rag_platform.reranker import CrossEncoderReranker
        cfg.reranker_model, cfg.reranker_revision, cfg.allow_model_download = args.reranker, args.reranker_revision, True
        reranker = CrossEncoderReranker(cfg)
        scores = reranker.model.predict([('How long is recovery?', 'The recovery target is seventeen minutes.')], show_progress_bar=False)
        manifest.update(reranker_model=args.reranker, reranker_revision=reranker.revision,
                        reranker_smoke_score=float(scores[0]), real_reranker_inference=True)
    for key in ['embedding_revision'] + (['reranker_revision'] if args.both else []):
        if not re.fullmatch(r'[a-f0-9]{40,64}', manifest[key] or ''):
            raise RuntimeError('Could not resolve immutable model revision; pass an explicit revision')
    path = Path(__import__('os').environ['XDG_CACHE_HOME']) / 'model_versions.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Model revisions and inference checks saved:', path)


if __name__ == '__main__':
    main()
