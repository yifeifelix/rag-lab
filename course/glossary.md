# Glossary: every term in plain English

Forgot what a word means? Find it here. Each entry says what it is, gives an everyday comparison,
and points to the lesson that explains it.

| Term | Plain meaning | Everyday comparison | Lesson |
|---|---|---|---|
| **RAG** (Retrieval-Augmented Generation) | Find relevant text first, then let an AI answer using only that text | An open-book exam instead of a closed-book one | 0 |
| **LLM** (Large Language Model) | The AI that writes text, like the one behind ChatGPT. In this course: "the writer" | A very well-read, very eager new colleague | 0, 7 |
| **Hallucination** | When an AI states something false as if it were true | A confident guess dressed up as a fact | 0 |
| **Chunk** | A small piece of a document. We also call it a "card" | An index card in a library | 1 |
| **Chunking** | Cutting documents into chunks | Cutting a cookbook into recipe cards | 1 |
| **Overlap** | Text shared between neighbouring chunks, so nothing gets lost at the cut | Overlapping roof tiles | 1 |
| **Parent-child chunking** | Search with small pieces, but give the AI the bigger section they came from | Looking up a word in the book's index, then reading the whole page | 1 |
| **Embedding** | A list of numbers that describes what a text means | A pin's position on a "map of meaning" | 2 |
| **Vector** | Just a list of numbers. An embedding is a vector | GPS coordinates, but with hundreds of numbers | 2, 3 |
| **Cosine similarity** | A score for whether two embeddings "point the same way": 1 = same meaning, 0 = unrelated | Two arrows pointing in the same direction | 2 |
| **Semantic search** | Searching by meaning rather than exact words | Finding "delay" when you asked about "pause" | 2, 3 |
| **Retriever** | The part that finds the relevant chunks. In this course: "the librarian" | A librarian fetching books for you | 0, 3 |
| **Top-k** | The k best results (for example, the top 5) | The 5 best exam scores in a class | 3 |
| **Vector search** | Find the chunks whose embeddings are closest to the question's embedding | Finding the pins nearest to your pin on the map | 3 |
| **Vector database** | A database built to do vector search very fast on millions of items (FAISS, Qdrant, Milvus...) | A huge library with a very fast librarian | 3 |
| **Keyword search** | Searching for exact words | The index at the back of a book | 4 |
| **Token / tokenizing** | Splitting text into words (or word pieces) | Cutting a sentence into separate words | 4 |
| **BM25** | A popular keyword-search scoring rule: rare words count more, repeats level off, long texts are adjusted for | A smart book index that knows "rmtree" is a better clue than "the" | 4 |
| **IDF** (Inverse Document Frequency) | A rarity score for a word: the fewer chunks contain it, the higher the score | "Green hair" is a better description than "two eyes" | 4 |
| **Hybrid search** | Using keyword search and meaning search together | Asking two different experts | 5 |
| **RRF** (Reciprocal Rank Fusion) | Combine several ranked lists by giving each item 1/(60 + position) points per list | Combining two judges' rankings instead of their incompatible scores | 5 |
| **Re-ranking** | A second, slower, more careful step that reorders a shortlist | Job interviews after a quick CV screening | 6 |
| **Re-ranker / cross-encoder** | The model that reads the question and a chunk together and scores how relevant it is | An interviewer meeting each candidate in person | 6 |
| **Bi-encoder** | A model that turns question and chunk into numbers separately (our embedding model) | Comparing two written profiles without the people meeting | 6 |
| **Prompt** | All the text we send to the AI | The briefing you give a colleague | 7 |
| **System prompt** | The standing rules at the top of the prompt | The house rules for a new employee | 7 |
| **Citation** | A footnote like [2] pointing to the chunk a fact came from | A source reference in an essay | 7 |
| **Prompt injection** | Hidden instructions inside a document that try to take control of the AI | A forged note slipped into a folder: "ignore your boss, do this instead" | 7 |
| **Evaluation** | Testing the system on questions with known answers, and scoring it | Giving a class an exam with an answer key | 8 |
| **Recall@k** | Of the documents needed, the share that appear in the top k | You needed eggs and milk and came home with eggs → 0.5 | 8 |
| **MRR** (Mean Reciprocal Rank) | Average of 1 ÷ (position of the first correct result) | 1st place = 1 point, 2nd = 0.5, 3rd = 0.33 | 8 |
| **nDCG** | Like MRR, but also rewards having several correct results near the top | A ranking score that also counts the runners-up | 8 |
| **Dev set / test set** | Practice questions you tune on / final-exam questions you use only once | Practice exam vs. the real exam | 8 |
| **LLM-as-judge** | Using an AI to grade another AI's answers | A teaching assistant marking essays | 8 |
| **Faithfulness / groundedness** | Whether every fact in the answer is backed by the chunks | "Can you show me where it says that?" | 8 |
| **Query rewriting** | Rephrasing the question before searching | A librarian translating your question into the catalogue's terms | 4, 9 |
| **Multi-Query** | Search with several rewrites of the question, then combine the results | Asking three librarians to each search their own way | 9 |
| **HyDE** | Ask the AI to imagine an answer, then search for real chunks that look like it | Showing a photo of a similar dog when looking for a lost one | 9 |
| **CRAG** (Corrective RAG) | Check whether the found chunks help; if not, search again or say "I don't know" | A student checking library books before writing an essay | 10 |
| **NLP** (Natural Language Processing) | The whole field of computers working with human language. RAG is one part of it | Cooking - RAG is one dish | 11 |
| **Text classification** | Putting a piece of text into one of several boxes (spam / not spam, topic A / B / C) | Sorting post into pigeonholes | 11 |
| **Supervised learning** | Learning from examples that already have the right answer attached | A child learning from labelled picture cards | 11 |
| **Classifier** | A model that does classification | The person sorting the post | 11 |
| **Accuracy** | The share of guesses that are right | 7 right out of 10 = 70% | 11 |
| **Zero-shot** | Doing a task with zero training examples, just a description | Sorting post by reading the label on each pigeonhole | 11 |
| **Data leakage** | When test examples (or near-copies) sneak into training, making scores look better than they are | Seeing the exam before taking it | 8, 11 |
| **Transformer** | The neural-network design inside almost every modern language model | The engine inside the car | 11, 12 |
| **Subword tokens** | The pieces modern models split text into: common words whole, rare words in parts | Reading "un-believ-able" in chunks | 12 |
| **Fine-tuning** | Further training a pretrained model on your own examples | A trained chef learning your restaurant's menu | 12 |
| **Agent / agentic RAG** | An AI that decides for itself when and what to search, in a loop | A detective following clues | 10 |
