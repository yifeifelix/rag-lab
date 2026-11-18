"""Real Qdrant transport check with explicit numeric vectors, not model evaluation."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import numpy as np

from rag_platform.chunking import Chunk
from rag_platform.config import RAGConfig
from rag_platform.runtime import configure_runtime
from rag_platform.vector_store import QdrantDenseIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:6333')
    args = parser.parse_args()
    configure_runtime()
    indices = []
    collection = None
    with tempfile.TemporaryDirectory(dir=Path(__import__('os').environ['TMPDIR'])) as directory:
        cfg = RAGConfig(corpus_path=Path(directory)/'corpus.jsonl', qdrant_url=args.url,
                        embedding_backend='sentence_transformers', vector_backend='qdrant',
                        qdrant_collection='rag_contract_' + uuid.uuid4().hex[:12])
        encoder = SimpleNamespace(embedding_dim=4, revision='numeric-contract-fixture-v1')
        chunks = [Chunk('a','a','Numeric A','EU','numeric fixture a','numeric fixture a'),
                  Chunk('b','b','Numeric B','US','numeric fixture b','numeric fixture b')]
        vectors = np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        try:
            first = QdrantDenseIndex(chunks,encoder,vectors,cfg)
            indices.append(first)
            collection = first.collection
            first.activate()
            assert first.upserted_points == 2
            assert first.search_vector(vectors[0],2,'EU')[0][0] == 0
            modified = [Chunk('a2','a','Numeric A v2','EU','changed fixture a','changed fixture a'), chunks[1]]
            second = QdrantDenseIndex(modified,encoder,vectors,cfg)
            indices.append(second)
            assert second.upserted_points == 1
            first.retire()
            second.activate()
            assert first.search_vector(vectors[0],2)[0][0] == 0
            restart = QdrantDenseIndex(modified,encoder,vectors,cfg)
            indices.append(restart)
            assert restart.upserted_points == 0
            deleted = QdrantDenseIndex([chunks[1]],encoder,vectors[1:],cfg)
            indices.append(deleted)
            second.retire()
            restart.retire()
            deleted.activate()
            assert len(deleted._all_ids()) == 1
            assert deleted.search_vector(vectors[1],2,'EU') == []
            print(json.dumps({'status':'passed', 'backend':'real Qdrant REST',
                'vectors':'explicit numeric fixture; no learned model or quality claim',
                'checks':['upsert','query','domain filter','incremental update','retired snapshot','restart reuse','delete']}))
        finally:
            if collection:
                with httpx.Client(base_url=args.url,timeout=15,trust_env=False) as client:
                    response = client.delete('/collections/' + collection)
                    response.raise_for_status()
            for index in indices:
                index.close()


if __name__ == '__main__':
    main()
