---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1 · Documents and chunking

**Time:** 45 minutes · **You will:** implement a fixed-size chunker and compare three chunking strategies.

## Why split documents?

Our `typing` page is ~24,000 characters. If we stored it as one item:

- the embedding model would squash 24k characters into one vector - the details blur together;
- a search for "What does typing.cast do?" would return the *whole page*, and the LLM would pay for (and get distracted by) 23,900 irrelevant characters.

So we split documents into **chunks**. The art is choosing the size:

| Chunks too small | Chunks too big |
|---|---|
| precise matches, but lose context ("it returns *it*" - what is *it*?) | context kept, but the signal is diluted and prompts get expensive |

```{code-cell} ipython3
from raglab import load_documents
docs = load_documents("python_docs")
doc = next(d for d in docs if d.doc_id == "py.heapq")
print(len(doc.text), "characters")
print(doc.text[:600])
```

## Strategy 1 · Fixed-size windows

The simplest idea: cut every `size` characters. To avoid cutting a sentence in half with no
second chance, consecutive windows **overlap**:

```text
text:     abcdefghijklmnopqrstuvwxyz
size=10, overlap=4  (step = size - overlap = 6)
window 1: abcdefghij
window 2:       ghijklmnop
window 3:             mnopqrstuv
window 4:                   stuvwxyz   ← reaches the end, stop
```

### ✏️ Exercise 1.1 - implement it

Return a list of strings. Rules: each window has at most `size` characters, starts `size - overlap`
after the previous one, and you stop as soon as a window reaches the end of the text.

```{code-cell} ipython3
def fixed_size_chunks(text: str, size: int, overlap: int) -> list[str]:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_fixed_size_chunker
check_fixed_size_chunker(fixed_size_chunks)
```

Look at what it does to real text - see the words cut in half at the edges?

```{code-cell} ipython3
from raglab.chunking import fixed_size_chunks as reference_fixed  # the library version
for piece in reference_fixed(doc.text, 300, 50)[:3]:
    print(repr(piece), "\n")
```

## Strategy 2 · Recursive splitting on natural boundaries

Better: split where the *author* put boundaries. Try headings first; if a piece is still too big,
split it by paragraphs, then lines, then sentences, then words. Small neighbours are merged back.
This is what LangChain's popular `RecursiveCharacterTextSplitter` does.

```{code-cell} ipython3
from raglab.chunking import chunk_documents
from raglab.display import show_chunks

recursive = chunk_documents([doc], method="recursive", max_chars=500)
show_chunks(recursive, n=4)
```

## Strategy 3 · Parent-child ("small to big")

Search with **small** chunks (precise matching) but hand the LLM the **bigger section** they came
from (enough context). Each `Chunk` has `.text` (what we search) and `.context` (what the LLM reads).

```{code-cell} ipython3
pc = chunk_documents([doc], method="parent_child", max_chars=200)
c = pc[3]
print("SEARCHED TEXT:\n", c.text, "\n\nCONTEXT GIVEN TO THE LLM:\n", c.context[:700])
```

## Compare the strategies on the whole corpus

```{code-cell} ipython3
import pandas as pd

rows = []
for method, kwargs in [("fixed", dict(size=500, overlap=100)), ("recursive", dict(max_chars=500)),
                       ("section", {}), ("parent_child", dict(max_chars=300))]:
    chunks = chunk_documents(docs, method=method, **kwargs)
    lengths = pd.Series([len(c.text) for c in chunks])
    rows.append({"method": method, "chunks": len(chunks), "median chars": int(lengths.median()),
                 "max chars": lengths.max()})
pd.DataFrame(rows).set_index("method")
```

### ✏️ Exercise 1.2 - think like a RAG engineer

In the cell below, write 2-3 sentences: `section` chunks have a very large max size. Which
document causes it, and what problem might that create for retrieval or for the LLM's prompt?
(No checker - compare with the solutions page afterwards.)

```{code-cell} ipython3
# Find the biggest section chunk here, then write your answer as a comment.
```

## Key takeaways

- Chunking decides *what can be found*. No later step can recover information that was cut apart.
- Overlap is a cheap insurance policy against boundary cuts.
- Respect document structure when you have it (headings, paragraphs).
- Parent-child decouples "what to match" from "what to read".

We'll **measure** which strategy retrieves best in lesson 8 - never pick one on gut feeling alone.

## Going further

- RAG_Techniques: [choose_chunk_size](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/choose_chunk_size.ipynb), [semantic_chunking](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/semantic_chunking.ipynb), [proposition_chunking](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/proposition_chunking.ipynb), [contextual_chunk_headers](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb) · local: `references/NirDiamant_RAG_Techniques/all_rag_techniques/`
- all-in-rag (Chinese), chapter 2 - data loading and chunking: `references/datawhalechina_all-in-rag/docs/chapter2/`
