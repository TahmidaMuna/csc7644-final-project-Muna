"""
corpus_loader.py
----------------
Loads and chunks PDF methodology documents for the RAG component.

Supported documents (place in data/corpus/):
    - fema_nri_technical_documentation.pdf
    - cdc_svi_methodology.pdf
    - cejst_methodology.pdf

Uses PyMuPDF (fitz) for text extraction and splits on sentence boundaries
to produce clean, retrievable chunks.
"""

import os
import re
from pathlib import Path
from typing import Optional


# Target chunk size in characters (roughly 200-300 tokens)
CHUNK_SIZE = 1000

# Overlap between consecutive chunks to preserve context across boundaries
CHUNK_OVERLAP = 150


def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Falls back to a placeholder string if PyMuPDF is not installed,
    allowing the system to run in a degraded (no-RAG) mode.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        Full extracted text of the document.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415

        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text)
    except ImportError:
        print(
            f"[Warning] PyMuPDF not installed. Cannot extract text from {pdf_path}. "
            "Run: pip install pymupdf"
        )
        return ""
    except Exception as exc:  # noqa: BLE001
        print(f"[Warning] Failed to extract text from {pdf_path}: {exc}")
        return ""


def _chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a long text string into overlapping chunks with source metadata.

    Splits preferentially at sentence boundaries (period + space) to avoid
    cutting mid-sentence. Each chunk is tagged with its source document name
    for citation purposes.

    Parameters
    ----------
    text : str
        Full document text to split.
    source : str
        Human-readable name for the source document.
    chunk_size : int
        Target character length for each chunk.
    overlap : int
        Number of characters to repeat between adjacent chunks.

    Returns
    -------
    list[dict]
        Each item has keys: 'text' (str), 'source' (str), 'chunk_id' (int).
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to split at a sentence boundary within the last 200 chars of the window
        if end < len(text):
            search_start = max(start + chunk_size - 200, start)
            boundary = text.rfind(". ", search_start, end)
            if boundary != -1:
                end = boundary + 1  # include the period

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

        # Move forward by chunk_size minus overlap
        start = max(end - overlap, start + 1)

    return chunks


def load_corpus(corpus_dir: str) -> list[dict]:
    """
    Load all PDF files from a directory, extract text, and chunk them.

    Also loads any plain-text (.txt) fallback files in the same directory,
    which allows the RAG component to work even when PDFs are unavailable
    (e.g., using pre-extracted text snippets for testing).

    Parameters
    ----------
    corpus_dir : str
        Path to the directory containing methodology documents.

    Returns
    -------
    list[dict]
        Flat list of chunk dicts: {'text': str, 'source': str, 'chunk_id': int}.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[Warning] Corpus directory {corpus_dir} does not exist. RAG will be empty.")
        return []

    all_chunks = []

    # Process PDF files
    for pdf_file in sorted(corpus_path.glob("*.pdf")):
        source_name = pdf_file.stem.replace("_", " ").title()
        print(f"[Corpus] Loading: {pdf_file.name}")
        raw_text = _extract_text_from_pdf(str(pdf_file))
        if raw_text.strip():
            chunks = _chunk_text(raw_text, source=source_name)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks created from {len(raw_text)} characters")

    # Process plain-text fallback files
    for txt_file in sorted(corpus_path.glob("*.txt")):
        source_name = txt_file.stem.replace("_", " ").title()
        print(f"[Corpus] Loading text fallback: {txt_file.name}")
        raw_text = txt_file.read_text(encoding="utf-8", errors="replace")
        if raw_text.strip():
            chunks = _chunk_text(raw_text, source=source_name)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks created")

    print(f"[Corpus] Total chunks loaded: {len(all_chunks)}")
    return all_chunks
