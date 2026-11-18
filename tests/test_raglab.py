"""Fast tests that need no model downloads and no LLM."""

import numpy as np
import pytest

from raglab import checks, load_documents, load_qa
from raglab.chunking import chunk_documents, fixed_size_chunks, recursive_chunks, section_chunks
from raglab.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank, evaluate_retriever
from raglab.retrieval import BM25Retriever, reciprocal_rank_fusion, tokenize, top_k_indices
from raglab.rag import cited_numbers


@pytest.fixture(scope="module")
def docs():
    return load_documents("python_docs")


def test_reference_implementations_pass_their_checkers():
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    idf = lambda n, df: float(np.log(1 + (n - df + 0.5) / (df + 0.5)))
    term = lambda tf, idf, dl, al, k1=1.5, b=0.75: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / al))
    assert checks.check_fixed_size_chunker(fixed_size_chunks)
    assert checks.check_cosine_similarity(cos)
    assert checks.check_top_k(lambda s, k: list(top_k_indices(s, k)))
    assert checks.check_idf(idf)
    assert checks.check_bm25_term(term)
    assert checks.check_rrf(reciprocal_rank_fusion)
    assert checks.check_recall_at_k(recall_at_k)
    assert checks.check_mrr(reciprocal_rank)


def test_checker_reports_unimplemented(capsys):
    def todo(*_):
        raise NotImplementedError
    assert not checks.check_rrf(todo)
    assert "not implemented" in capsys.readouterr().out


def test_qa_spans_exist_in_corpus(docs):
    text = {d.doc_id: d.text for d in docs}
    for item in load_qa("python_docs"):
        for doc_id, span in zip(item["relevant_doc_ids"], item["supporting_spans"]):
            assert span in text[doc_id], (item["qid"], span)
    policy_ids = {d.doc_id for d in load_documents("policies")}
    for item in load_qa("policies"):
        assert set(item["relevant_doc_ids"]) <= policy_ids


def test_chunkers_respect_limits(docs):
    for doc in docs[:10]:
        assert all(len(p) <= 800 for p in recursive_chunks(doc.text, 800))
        assert "".join(section_chunks(doc.text)).replace("\n", "") != ""
    parents = chunk_documents(docs[:3], "parent_child", max_chars=300)
    assert all(len(c.text) <= 300 and c.text in c.context for c in parents)


def test_tokenize_keeps_dotted_names():
    assert tokenize("How does json.dumps work?") == ["json.dumps", "json", "dumps", "work"]


def test_bm25_finds_exact_identifiers(docs):
    bm25 = BM25Retriever(chunk_documents(docs, "section"))
    assert bm25.search("shutil.rmtree", 3)[0].doc_id == "py.shutil"
    results = evaluate_retriever(bm25, load_qa("python_docs"), k=5, name="bm25")
    assert results["recall@5"].mean() > 0.5


def test_metrics_and_citations():
    assert ndcg_at_k(["a", "b"], ["a"], 5) == 1.0
    assert ndcg_at_k(["b", "a"], ["a"], 5) < 1.0
    assert cited_numbers("Use dumps [1][3] and [1].") == [1, 3]
