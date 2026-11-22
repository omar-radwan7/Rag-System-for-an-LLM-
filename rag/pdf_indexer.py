#!/usr/bin/env python3
"""
pdf_indexer.py — Uploaded PDFs -> clean text -> overlapping chunks -> vectors ->
in-memory FAISS index.

Everything stays in RAM: the documents are never written to disk, and the index
is gone when the session ends.
"""

import io
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """Raw text of every page, as a list of (page_number, text)."""
    from pypdf import PdfReader
    from rag.arabic_fix import fix_arabic_ligatures

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for n, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((n, fix_arabic_ligatures(text)))
    return pages


def normalize_text(text: str) -> str:
    """
    Regex cleanup of extracted PDF text: page numbers on their own line,
    words split by a hyphen at a line end, broken lines inside a paragraph,
    and repeated spaces.
    """
    text = text.replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"(?m)^\s*(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?\s*$", "", text, flags=re.I)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", " ", text)        # keep paragraph breaks
    text = re.sub(r"\s*\n\s*", " ", text)           # join broken lines
    text = text.replace(" ", "\n\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def find_headings(raw: str) -> list:
    """
    Heading-like lines from raw page text: short (at most 8 words), no closing
    punctuation, starting with a number or a capital letter (or a non-Latin
    letter). Used to give the model a table of contents of the whole document.
    """
    headings = []
    for line in raw.splitlines():
        line = line.strip()
        words = line.split()
        if not (3 <= len(line) <= 70 and 1 <= len(words) <= 8):
            continue
        if line[-1] in ".,;:!?" or not re.search(r"[^\W\d_]", line):
            continue
        first = line[0]
        if first.isdigit() or first.isupper() or (first.isalpha() and not first.islower() and not first.isupper()):
            headings.append(line)
    return headings


def chunk_text(text: str, size: int = 500, overlap: int = 150) -> list:
    """
    Sliding window over characters: chunks of about `size` characters, each
    sharing `overlap` characters with the previous one so a sentence cut at a
    border is still complete in one of them. Borders are moved back to the
    nearest space so words are never cut.
    """
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def build_memory_index(files: list) -> dict:
    """
    files: list of (file_name, pdf_bytes).
    Returns {'index', 'metadata', 'chunk_count', 'doc_names'}; pass it to the
    pipeline as dynamic_index.
    """
    import faiss
    from rag.models import passage_text
    from rag.rag_pipeline import _get_embed_model

    cfg = _load_config()
    size = cfg.get("chunk_size_chars", 500)
    overlap = cfg.get("chunk_overlap_chars", 150)

    metadata, outline = [], []
    for name, pdf_bytes in files:
        headings = []
        for page, raw in extract_text_from_pdf(pdf_bytes):
            headings.extend(h for h in find_headings(raw) if h not in headings)
            for piece in chunk_text(normalize_text(raw), size, overlap):
                metadata.append({
                    "chunk_id": len(metadata),
                    "source": f"{name}, p. {page}",
                    "text": piece,
                })
        if headings:
            outline.append(f"{name}: " + " | ".join(headings[:60]))

    if not metadata:
        raise ValueError("No text could be extracted. The PDF may be a scanned image.")

    model = _get_embed_model()
    texts = [passage_text(model.rag_name, c["text"]) for c in metadata]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=16,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "index": index,
        "metadata": metadata,
        "chunk_count": len(metadata),
        "outline": "\n".join(outline),
        "text_chars": sum(len(c["text"]) for c in metadata),
        "doc_names": ", ".join(name for name, _ in files),
    }
