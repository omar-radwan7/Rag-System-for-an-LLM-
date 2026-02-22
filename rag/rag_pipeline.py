#!/usr/bin/env python3
"""
rag_pipeline.py — Full RAG pipeline: embed query → retrieve from FAISS → prompt LLM.
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAG_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
FAISS_INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
METADATA_PATH = os.path.join(RAG_DIR, "chunk_metadata.json")

# Lazy-loaded globals
_embed_model = None
_faiss_index = None
_metadata = None


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        cfg = _load_config()
        model_name = cfg.get("embedding_model", "intfloat/multilingual-e5-small")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    return _faiss_index


def _get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    return _metadata


def retrieve(query: str, top_k: int = 5) -> tuple:
    """
    Embed query and retrieve top_k most similar chunks.
    Returns (chunks_list, retrieval_time_seconds).
    """
    model = _get_embed_model()
    index = _get_faiss_index()
    metadata = _get_metadata()

    start = time.time()
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, indices = index.search(q_emb, top_k)
    retrieval_time = round(time.time() - start, 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            chunk = metadata[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

    return results, retrieval_time


# Arabic character name & concept aliases found in the book
_NAME_ALIASES = {
    "المؤلف": ["عمرو عبد الحميد"],
    "الكاتب": ["عمرو عبد الحميد"],
}

def _expand_query(query: str) -> list:
    """Returns a list of search queries: original + alias expansions."""
    queries = [query]
    for name, aliases in _NAME_ALIASES.items():
        if name in query:
            for alias in aliases:
                queries.append(query.replace(name, alias))
    return queries


def retrieve_multi(query: str, top_k: int = 5) -> tuple:
    """
    Search using multiple query variants (aliases) and merge + deduplicate results.
    Returns (chunks_list, retrieval_time).
    """
    queries = _expand_query(query)
    seen_ids = set()
    all_chunks = []
    start = time.time()

    for q in queries:
        chunks, _ = retrieve(q, top_k=top_k)
        for c in chunks:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                all_chunks.append(c)

    # Sort by score descending and take the best top_k
    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    retrieval_time = round(time.time() - start, 3)
    return all_chunks[:top_k], retrieval_time


CHARACTER_FACTS = ""


def build_rag_prompt(question: str, chunks: list, history: list = None) -> str:
    """Build the Arabic RAG prompt with context and optional history."""
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    
    hist_str = ""
    if history and len(history) > 0:
        hist_str = "سجل المحادثة السابقة (استخدمه لفهم السياق والضمائر في السؤال الحالي):\n"
        for msg in history[-6:]:  # keep last 3 turns
            role = "المستخدم" if msg["role"] == "user" else "الذكاء الاصطناعي"
            hist_str += f"{role}: {msg['content']}\n"
        hist_str += "\n"

    prompt = f"""أنت مساعد ذكي وخبير باللغة العربية الفصحى. مهمتك هي الإجابة على السؤال بدقة متناهية بناءً على السياق المتاح فقط.
تعليمات:
1. صغ الإجابة بعناية باللغة العربية الفصحى.
2. لا تخلق أو تخترع أي معلومات من خارج النص.
3. إذا كان السياق لا يحتوي على إجابة، قل حصراً: "لا يوجد في النص".
4. كن مختصراً.
{CHARACTER_FACTS}
{hist_str}السياق المتاح من الكتاب:
{context}

السؤال الحالي:
{question}

الإجابة باللغة العربية:"""
    return prompt


def ask(question: str, model: str = None, top_k: int = 2, history: list = None) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.
    """
    from rag.ollama_client import generate

    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    # Retrieve: always prepend recent history topic to query to improve context recall
    search_query = question
    if history and len(history) >= 2 and len(question.split()) < 12:
        # Combine with the previous user question for richer semantic search
        search_query = f"{history[-2]['content']} {question}"
        
    chunks, retrieval_time = retrieve_multi(search_query, top_k=top_k)
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Build prompt and generate
    rag_prompt = build_rag_prompt(question, chunks, history)
    start = time.time()
    result = generate(model, rag_prompt, temperature=0.1, max_tokens=150)
    generation_time = round(time.time() - start, 3)

    return {
        "answer": result["text"],
        "sources": chunk_ids,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "model": model,
        "chunks": chunks,
    }

def ask_stream(question: str, model: str = None, top_k: int = 2, history: list = None):
    from rag.ollama_client import generate_stream

    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    # Retrieve: always prepend recent history topic to query to improve context recall
    search_query = question
    if history and len(history) >= 2 and len(question.split()) < 12:
        search_query = f"{history[-2]['content']} {question}"
        
    chunks, retrieval_time = retrieve_multi(search_query, top_k=top_k)
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Build prompt and generate stream
    rag_prompt = build_rag_prompt(question, chunks, history)
    
    start = time.time()
    stream = generate_stream(model, rag_prompt, temperature=0.1, max_tokens=150)
    
    return stream, chunk_ids, retrieval_time, start


def ask_llm_only(question: str, model: str = None) -> dict:
    from rag.ollama_client import generate
    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    prompt = f"""أجب بشكل مختصر جدا (لا تتجاوز 100 كلمة):\n\nالسؤال:\n{question}\n\nالإجابة:"""
    start = time.time()
    result = generate(model, prompt, temperature=0.3, max_tokens=150)
    generation_time = round(time.time() - start, 3)

    return {
        "answer": result["text"],
        "sources": [],
        "retrieval_time": 0,
        "generation_time": generation_time,
        "model": model,
    }

def ask_llm_only_stream(question: str, model: str = None):
    from rag.ollama_client import generate_stream
    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    prompt = f"""أجب بشكل مختصر جدا (لا تتجاوز 100 كلمة):\n\nالسؤال:\n{question}\n\nالإجابة:"""
    start = time.time()
    stream = generate_stream(model, prompt, temperature=0.3, max_tokens=150)
    
    return stream, [], 0, start

if __name__ == "__main__":
    q = "ما هي الفكرة الرئيسية في الكتاب؟"
    print("RAG answer:")
    result = ask(q)
    print(result["answer"][:300])
    print(f"Sources: {result['sources']}")
    print(f"Retrieval: {result['retrieval_time']}s, Generation: {result['generation_time']}s")
