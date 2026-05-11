"""
retriever.py
------------
Implements the RAG retrieval component using FAISS for vector similarity search
and OpenAI's text-embedding-3-small for chunk embeddings.

The index is built lazily on first retrieval and cached in memory for the
duration of the session. Re-indexing is triggered automatically if the
corpus directory contents change.
"""

import os
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

from rag.corpus_loader import load_corpus


# Index cache file stored alongside the corpus for reuse between sessions
CACHE_FILENAME = ".rag_index_cache.pkl"

# Embedding model — small and cheap, appropriate for short methodology chunks
EMBEDDING_MODEL = "text-embedding-3-small"

# Number of top chunks to return per retrieval query
DEFAULT_TOP_K = 3

_retriever_instance: Optional["RAGRetriever"] = None


def _compute_corpus_hash(corpus_dir: str) -> str:
    """
    Compute a hash of all file modification times in the corpus directory.
    Used to detect when the corpus has changed and the index needs rebuilding.

    Parameters
    ----------
    corpus_dir : str
        Path to the corpus directory.

    Returns
    -------
    str
        MD5 hash string representing the current state of the corpus.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        return "empty"

    file_stats = []
    for f in sorted(corpus_path.iterdir()):
        if f.suffix in {".pdf", ".txt"}:
            file_stats.append(f"{f.name}:{f.stat().st_mtime}")

    return hashlib.md5(":".join(file_stats).encode()).hexdigest()


class RAGRetriever:
    """
    Retrieval-Augmented Generation component backed by a FAISS vector index.

    On first use, loads the corpus from `corpus_dir`, embeds all chunks using
    OpenAI's embedding API, and builds a FAISS flat L2 index. Subsequent calls
    reuse the in-memory index or load from a pickle cache if available.

    Attributes
    ----------
    corpus_dir : str
        Directory containing methodology PDF/text files.
    openai_api_key : str
        OpenAI API key for embedding calls.
    _index : faiss.IndexFlatL2 or None
        In-memory FAISS index (None until first retrieval).
    _chunks : list[dict] or None
        Ordered list of corpus chunks parallel to the FAISS index.
    """

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the retriever. Does not build the index until first retrieval call.

        Parameters
        ----------
        corpus_dir : str
            Path to corpus directory.
        openai_api_key : str, optional
            OpenAI API key. Falls back to OPENAI_API_KEY env variable.
        """
        self.corpus_dir = corpus_dir
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._index = None
        self._chunks: Optional[list[dict]] = None

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings using OpenAI's API.

        Processes texts in batches of 100 to stay within API limits.

        Parameters
        ----------
        texts : list[str]
            Strings to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors in the same order as input texts.
        """
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=self.openai_api_key)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _build_index(self) -> None:
        """
        Build the FAISS index from scratch.

        Loads the corpus, embeds all chunks, and stores the index in memory.
        Saves a pickle cache to disk for fast reuse in subsequent sessions.
        Gracefully degrades to keyword-only search if FAISS is not installed.
        """
        chunks = load_corpus(self.corpus_dir)
        self._chunks = chunks

        if not chunks:
            print("[RAG] No corpus chunks found. Retrieval will return empty results.")
            return

        try:
            import faiss  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:
            print(
                "[RAG] FAISS not installed. Falling back to keyword search. "
                "Run: pip install faiss-cpu"
            )
            self._index = None
            return

        if not self.openai_api_key:
            print("[RAG] No OpenAI API key — skipping embedding, using keyword fallback.")
            self._index = None
            return

        print(f"[RAG] Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
        texts = [c["text"] for c in chunks]
        embeddings = self._get_embeddings(texts)

        vectors = np.array(embeddings, dtype="float32")
        dim = vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        self._index = index

        # Cache to disk
        cache_path = Path(self.corpus_dir) / CACHE_FILENAME
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"chunks": chunks, "vectors": vectors, "hash": _compute_corpus_hash(self.corpus_dir)}, f)
            print(f"[RAG] Index cached to {cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] Warning: Could not write cache: {exc}")

        print(f"[RAG] Index built: {index.ntotal} vectors, dim={dim}")

    def _load_cached_index(self) -> bool:
        """
        Attempt to load a previously cached FAISS index from disk.

        Validates that the cache matches the current corpus state before loading.

        Returns
        -------
        bool
            True if a valid cache was loaded, False otherwise.
        """
        cache_path = Path(self.corpus_dir) / CACHE_FILENAME
        if not cache_path.exists():
            return False

        try:
            import faiss  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            with open(cache_path, "rb") as f:
                cache = pickle.load(f)

            current_hash = _compute_corpus_hash(self.corpus_dir)
            if cache.get("hash") != current_hash:
                print("[RAG] Corpus changed — rebuilding index.")
                return False

            vectors = cache["vectors"]
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)

            self._index = index
            self._chunks = cache["chunks"]
            print(f"[RAG] Loaded cached index: {index.ntotal} vectors")
            return True

        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] Cache load failed: {exc}. Rebuilding.")
            return False

    def _keyword_fallback(self, query: str, top_k: int) -> list[str]:
        """
        Simple keyword-based retrieval fallback when FAISS is unavailable.

        Scores each chunk by the number of query terms it contains.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of chunks to return.

        Returns
        -------
        list[str]
            Top matching chunk texts.
        """
        if not self._chunks:
            return []

        query_terms = set(query.lower().split())
        scores = []
        for i, chunk in enumerate(self._chunks):
            chunk_words = set(chunk["text"].lower().split())
            score = len(query_terms & chunk_words)
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                chunk = self._chunks[idx]
                results.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

        return results

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """
        Retrieve the top-k most relevant corpus chunks for a given query.

        Uses FAISS vector similarity if available, falls back to keyword search.
        Builds or loads the index on first call.

        Parameters
        ----------
        query : str
            Natural language query describing the information needed.
        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        list[str]
            Formatted strings: "[Source: <name>]\\n<chunk text>".
        """
        # Lazy initialization
        if self._chunks is None:
            if not self._load_cached_index():
                self._build_index()

        # No corpus available
        if not self._chunks:
            return []

        # Use keyword fallback if FAISS index is not available
        if self._index is None:
            return self._keyword_fallback(query, top_k)

        try:
            import faiss  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            # Embed the query
            query_embedding = self._get_embeddings([query])[0]
            query_vector = np.array([query_embedding], dtype="float32")

            # Search the FAISS index
            distances, indices = self._index.search(query_vector, top_k)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    results.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

            return results

        except Exception as exc:  # noqa: BLE001
            print(f"[RAG] FAISS search failed: {exc}. Using keyword fallback.")
            return self._keyword_fallback(query, top_k)


def retrieve_context(query: str, k: int = 2, corpus_dir: str = "data/corpus") -> str:
    """
    Retrieve a compact methodology context block for injection into agent prompts.

    Reuses a module-level singleton retriever to avoid rebuilding the FAISS index
    on repeated calls within the same process. Returns at most two chunks and
    truncates each to 300 characters to keep downstream LLM prompt sizes bounded.

    Parameters
    ----------
    query : str
        Natural language query describing the context needed (e.g., score interpretation).
    k : int
        Maximum number of chunks to include; capped at 2 internally.
    corpus_dir : str
        Path to the corpus directory used to initialize the retriever.

    Returns
    -------
    str
        Newline-separated context block of retrieved chunks, each prefixed with
        its source document name. Returns an empty string if the corpus is empty.
    """
    global _retriever_instance

    if _retriever_instance is None or _retriever_instance.corpus_dir != corpus_dir:
        _retriever_instance = RAGRetriever(corpus_dir=corpus_dir)

    results = _retriever_instance.retrieve(query, top_k=min(k, 2))
    return "\n\n".join(result[:300] for result in results)
