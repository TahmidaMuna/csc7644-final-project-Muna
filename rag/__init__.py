"""
rag package
-----------
Retrieval-Augmented Generation components for the CSR Allocation Agent.
Includes PDF corpus loading, chunking, embedding, and FAISS-based retrieval.

Public API
----------
RAGRetriever
    Wraps a FAISS flat-L2 index built from methodology PDFs in ``data/corpus/``.
    Exposes a ``retrieve(query, top_k)`` method that returns ranked text chunks
    for injection into the agent's conversation context.
"""

from rag.retriever import RAGRetriever

__all__ = ["RAGRetriever"]
