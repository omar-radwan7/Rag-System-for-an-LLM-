#!/usr/bin/env python3
"""
rag_pipeline.py — Full RAG pipeline: embed query → retrieve from FAISS → prompt LLM.
"""

import json
import os
import re
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

# System prompt to ensure Arabic output without breaking whitespace generation
SYSTEM_PROMPT = "أجب دائماً باللغة العربية الفصحى الواضحة."


def clean_arabic_text(text: str) -> str:
    """Remove non-Arabic characters from full (non-streaming) model output."""
    cleaned = re.sub(
        r'[\u2E80-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]',
        '', text
    )
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()  # OK to strip for full text


def clean_chunk(chunk: str) -> str:
    """Remove non-Arabic characters from a streaming chunk WITHOUT stripping whitespace.
    
    IMPORTANT: Do NOT call .strip() here — each chunk may end with a space
    that separates it from the next word. Stripping removes that space and
    fuses words together (e.g. 'وصل ' + 'البطل' → 'وصلالبطل').
    """
    cleaned = re.sub(
        r'[\u2E80-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]',
        '', chunk
    )
    # Only collapse consecutive spaces — preserve single spaces
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned  # NO strip()


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
    "البطل": ["خالد", "خالد حسني"],
    "الشخصية الرئيسية": ["خالد", "خالد حسني"],
    "بطل القصة": ["خالد", "خالد حسني"],
    "الجد": ["عبدو", "جد خالد"],
    "حبيبته": ["منى"],
    # How Khaled got to Ard Zikola — via the underground tunnel
    "أرض زيكولا": ["السرداب", "خالد السرداب"],
    "وصول": ["السرداب", "نزل السرداب"],
    "كيف وصل": ["السرداب", "نزل السرداب خالد"],
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


CHARACTER_FACTS = """حقائق أساسية عن الكتاب (استخدمها فقط عند الحاجة):
- عنوان الكتاب: أرض زيكولا
- المؤلف: عمرو عبد الحميد
- بطل القصة / الشخصية الرئيسية: خالد حسني
- حبيبة خالد: منى
- جد خالد: عبدو
- نوع القصة: رواية مغامرات وفانتازيا عربية
"""


def build_rag_prompt(question: str, chunks: list, history: list = None) -> str:
    """Build the Arabic RAG prompt with context and optional history."""
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    
    hist_str = ""
    if history and len(history) > 0:
        hist_str = "سجل المحادثة السابقة (استخدمه لفهم السياق والضمائر في السؤال الحالي):\n"
        for msg in history[-10:]:  # keep last 5 turns
            role = "المستخدم" if msg["role"] == "user" else "الذكاء الاصطناعي"
            hist_str += f"{role}: {msg['content']}\n"
        hist_str += "\n"

    prompt = f"""أنت مساعد ذكي متخصص في الإجابة عن أسئلة حول كتاب "أرض زيكولا" للكاتب عمرو عبد الحميد.

تعليمات صارمة يجب اتباعها حرفياً:
1. أجب فقط وحصرياً بناءً على السياق المقدم أدناه. لا تخترع أو تضف أي معلومة غير موجودة في السياق.
2. إذا كان السياق لا يحتوي على إجابة واضحة، قل بالضبط: "لا يوجد في النص المتاح إجابة على هذا السؤال."
3. أجب باللغة العربية الفصحى السليمة فقط.
4. لا تستخدم أي لغة أخرى غير العربية إطلاقاً.
5. كن دقيقاً في أسماء الشخصيات ولا تخلط بينها.
6. إذا كان السؤال يحتوي على ضمائر أو إشارات (مثل "هو"، "هذا الرجل"، "معه"، "ماذا فعل")، استخدم سجل المحادثة أدناه لفهم المقصود بالضمير ثم أجب بناءً على السياق.

{CHARACTER_FACTS}
{hist_str}السياق المتاح من الكتاب:
{context}

السؤال الحالي:
{question}

الإجابة (بناءً على السياق وسجل المحادثة):"""
    return prompt


def _build_followup_query(question: str, history: list) -> str:
    """
    Build a richer search query for follow-up questions by including context
    from both the previous user question AND the assistant's answer.
    """
    if not history or len(history) < 2:
        return question

    # Only enhance short questions (likely follow-ups with pronouns)
    if len(question.split()) >= 15:
        return question

    # Gather context from recent history
    last_user_q = ""
    last_assistant_a = ""

    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant_a:
            # Take first 100 chars of assistant answer as context
            last_assistant_a = msg["content"][:100]
        elif msg["role"] == "user" and not last_user_q:
            last_user_q = msg["content"]
        if last_user_q and last_assistant_a:
            break

    # Combine: previous question + key part of answer + current question
    parts = []
    if last_user_q:
        parts.append(last_user_q)
    if last_assistant_a:
        parts.append(last_assistant_a)
    parts.append(question)

    return " ".join(parts)


def ask(question: str, model: str = None, top_k: int = 2, history: list = None) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.
    """
    from rag.ollama_client import generate

    if model is None:
        cfg = _load_config()
        model = cfg.get("chosen_model") or cfg["models"][0]

    # Build enriched search query for follow-ups
    search_query = _build_followup_query(question, history)
    
    # Use more chunks for follow-up questions (short questions = likely follow-ups)
    effective_top_k = top_k + 1 if history and len(question.split()) < 10 else top_k
        
    chunks, retrieval_time = retrieve_multi(search_query, top_k=effective_top_k)
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Build prompt and generate
    rag_prompt = build_rag_prompt(question, chunks, history)
    start = time.time()
    result = generate(model, rag_prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=512)
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

    # Build enriched search query for follow-ups
    search_query = _build_followup_query(question, history)
    
    # Use more chunks for follow-up questions (short questions = likely follow-ups)
    effective_top_k = top_k + 1 if history and len(question.split()) < 10 else top_k
        
    chunks, retrieval_time = retrieve_multi(search_query, top_k=effective_top_k)
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Build prompt and generate stream
    rag_prompt = build_rag_prompt(question, chunks, history)
    
    start = time.time()
    raw_stream = generate_stream(model, rag_prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=512)
    
    return raw_stream, chunk_ids, retrieval_time, start


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
