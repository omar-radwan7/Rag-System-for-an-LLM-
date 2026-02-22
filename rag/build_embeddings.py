#!/usr/bin/env python3
"""
build_embeddings.py — Embed all chunks with a multilingual model
and store them in a FAISS index.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAG_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

CHUNKS_JSONL = os.path.join(DATA_DIR, "chunks.jsonl")
FAISS_INDEX = os.path.join(RAG_DIR, "faiss.index")
METADATA_JSON = os.path.join(RAG_DIR, "chunk_metadata.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    chunks = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_index():
    from sentence_transformers import SentenceTransformer
    import faiss

    cfg = load_config()
    embed_model_name = cfg.get("embedding_model", "intfloat/multilingual-e5-small")

    print(f"📦 Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    print("📄 Loading chunks...")
    chunks = load_chunks()
    texts = [f"passage: {c['text']}" for c in chunks]  # e5 format prefix

    print(f"🔢 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print(f"💾 Building FAISS index (dim={embeddings.shape[1]})...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX)

    # Save metadata
    metadata = []
    for c in chunks:
        metadata.append({
            "chunk_id": c["chunk_id"],
            "chapter": c.get("chapter", ""),
            "text": c["text"],
            "word_count": c.get("word_count", len(c["text"].split())),
        })
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ FAISS index saved to {FAISS_INDEX}")
    print(f"✅ Metadata saved to {METADATA_JSON}")
    print(f"   Chunks: {len(chunks)}, Dimensions: {dim}")


if __name__ == "__main__":
    build_index()
