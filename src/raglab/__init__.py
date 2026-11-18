"""raglab: small, readable building blocks for the RAG course.

Every module is written to be read. If a lesson asks you to implement
something, the reference version lives here, so peek only after trying!
"""

from .data import Document, load_documents, load_qa
from .chunking import Chunk, chunk_documents

__all__ = ["Document", "Chunk", "load_documents", "load_qa", "chunk_documents"]
__version__ = "2.0.0"
