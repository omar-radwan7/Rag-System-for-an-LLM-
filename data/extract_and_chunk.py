#!/usr/bin/env python3
"""
extract_and_chunk.py — Extract text from the book PDF, clean it,
and split into overlapping chunks saved as JSONL.
"""

import fitz  # PyMuPDF
import json
import re
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "book.pdf")
CLEAN_TXT = os.path.join(DATA_DIR, "book_clean.txt")
CHUNKS_JSONL = os.path.join(DATA_DIR, "chunks.jsonl")

# Chunking parameters
CHUNK_SIZE = 150      # smaller chunks = more precise retrieval
CHUNK_OVERLAP = 30   # overlap to avoid cutting mid-sentence

def extract_text(pdf_path: str) -> str:
    """Extract all text from PDF using PyMuPDF (fitz), which handles Arabic text correctly."""
    full_text = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text:
            full_text.append(text)
        if (i + 1) % 20 == 0:
            print(f"  Extracted {i+1}/{len(doc)} pages...")
    print(f"  Total pages extracted: {len(full_text)}")
    return "\n".join(full_text)


def clean_text(raw: str) -> str:
    """Clean extracted text: remove page numbers, headers, artifacts."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip lines that are just numbers (page numbers)
        if re.match(r'^\d{1,4}$', line):
            continue
        # Skip very short lines that look like headers/footers
        if len(line) < 3 and not any('\u0600' <= c <= '\u06FF' for c in line):
            continue
        # Remove repeated dashes/underscores/equals (formatting artifacts)
        if re.match(r'^[-_=*]{3,}$', line):
            continue
        # Remove common header/footer patterns
        if re.match(r'^(صفحة|الصفحة|page)\s*\d+', line, re.IGNORECASE):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r'  +', ' ', text)
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks based on word count."""
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        # Try to detect chapter from nearby text
        chapter = detect_chapter(chunk_text_str)

        chunks.append({
            "chunk_id": chunk_id,
            "chapter": chapter,
            "text": chunk_text_str,
            "word_count": len(chunk_words),
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def detect_chapter(text: str) -> str:
    """Try to detect chapter name/number from text."""
    # Look for common Arabic chapter markers
    patterns = [
        r'(الفصل\s+[\u0600-\u06FF]+)',
        r'(الباب\s+[\u0600-\u06FF]+)',
        r'(الجزء\s+[\u0600-\u06FF]+)',
        r'(الفصل\s+\d+)',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    return ""


def main():
    print("📖 Extracting text from PDF...")
    raw_text = extract_text(PDF_PATH)

    print("🧹 Cleaning text...")
    clean = clean_text(raw_text)
    with open(CLEAN_TXT, "w", encoding="utf-8") as f:
        f.write(clean)
    print(f"  Saved clean text to {CLEAN_TXT} ({len(clean)} chars, ~{len(clean.split())} words)")

    print("✂️  Chunking text...")
    chunks = chunk_text(clean)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Created {len(chunks)} chunks → {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
