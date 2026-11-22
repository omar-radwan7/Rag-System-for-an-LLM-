#!/usr/bin/env python3
"""
rag_pipeline.py — Retrieval and generation.

Question -> hybrid search (vectors + BM25, optional reranker) -> strict prompt
with the retrieved passages -> local model through Ollama.

The model may only answer from the attached documents. If the passages do not
contain the answer, it says so instead of using its own knowledge.
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.models import DEFAULT_EMBEDDING_MODEL, query_text

RAG_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Lazy-loaded globals
_embed_model = None
_bm25_cache = {}
_reranker = None

SYSTEM_PROMPT = (
    "You are a document assistant. You answer only from the document passages "
    "you are given, never from your own knowledge. You always reply in the same "
    "language as the user's question."
)

def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = _load_config().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(model_name)
        _embed_model.rag_name = model_name
    return _embed_model


def _get_reranker():
    """Cross-encoder that rescores (question, passage) pairs; None if not configured."""
    global _reranker
    if _reranker is None:
        name = _load_config().get("reranker_model")
        if not name:
            return None
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(name, max_length=512)
    return _reranker


def _get_bm25(metadata: list):
    """BM25 index over the chunk texts, built once per metadata list."""
    from rag.keyword_search import BM25
    key = id(metadata)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25([c["text"] for c in metadata])
    return _bm25_cache[key]


def retrieve(query: str, top_k: int = 5, dynamic_index: dict = None) -> tuple:
    """
    Hybrid search: vector ranking (cosine similarity, FAISS IndexFlatIP on
    normalized vectors) + BM25 keyword ranking, merged with Reciprocal Rank
    Fusion, then reordered by the reranker if one is configured.
    dynamic_index is the in-memory index of the uploaded documents
    (pdf_indexer.build_memory_index). Returns (chunks, retrieval_time_seconds).
    """
    from rag.keyword_search import reciprocal_rank_fusion

    if not dynamic_index:
        raise ValueError("No documents loaded. Upload a PDF first.")
    index, metadata = dynamic_index["index"], dynamic_index["metadata"]

    model = _get_embed_model()
    bm25 = _get_bm25(metadata)
    start = time.time()

    q_emb = model.encode([query_text(model.rag_name, query)], normalize_embeddings=True)
    _, dense = index.search(np.array(q_emb, dtype="float32"), min(100, index.ntotal))
    rankings = [
        [i for i in dense[0] if 0 <= i < len(metadata)],
        bm25.rank(query),
    ]
    fused = reciprocal_rank_fusion(rankings)
    ranked = sorted(fused, key=fused.get, reverse=True)
    scores = fused

    reranker = _get_reranker()
    if reranker is not None and ranked:
        candidates = ranked[:_load_config().get("rerank_candidates", 30)]
        rerank_scores = reranker.predict([(query, metadata[i]["text"]) for i in candidates])
        scores = dict(zip(candidates, (float(x) for x in rerank_scores)))
        ranked = sorted(candidates, key=scores.get, reverse=True)

    # Small documents fit in the prompt completely: send all chunks in document
    # order, so questions about the whole document see all of it
    if dynamic_index.get("text_chars", 1e9) <= _load_config().get("full_context_chars", 4000):
        ranked = sorted(range(len(metadata)), key=lambda i: metadata[i]["chunk_id"])
        top_k = len(ranked)
        scores = {i: scores.get(i, 0.0) for i in ranked}

    chunks = []
    for i in ranked[:top_k]:
        chunk = metadata[i].copy()
        chunk["score"] = round(scores[i], 4)
        chunks.append(chunk)

    return chunks, round(time.time() - start, 3)


def _last_turn(history: list) -> list:
    """
    Rolling window: only the latest question and answer are sent with a new
    question. Sending the whole chat made the prompt grow every turn and pushed
    the answer time on CPU to about 60 s.
    """
    if not history:
        return []
    turn = [m for m in history if m["role"] in ("user", "assistant")][-2:]
    return [m for m in turn if m.get("content")]


def _search_query(question: str, history: list) -> str:
    """Short follow-ups ("and the second one?") are searched together with the last question."""
    turn = _last_turn(history)
    last_q = next((m["content"] for m in reversed(turn) if m["role"] == "user"), "")
    if last_q and len(question.split()) < 8:
        return f"{last_q} {question}"
    return question


def build_rag_prompt(question: str, chunks: list, history: list = None,
                     doc_names: str = None, outline: str = None) -> str:
    """Strict prompt: answer only from the passages, otherwise say it is not there."""
    passages = []
    for n, c in enumerate(chunks, 1):
        src = f" ({c['source']})" if c.get("source") else ""
        passages.append(f"[{n}]{src}\n{c['text']}")
    context = "\n\n".join(passages) if passages else "(no passages)"

    hist_str = ""
    turn = _last_turn(history)
    if turn:
        hist_str = "Previous turn (only to understand follow-up questions):\n"
        for m in turn:
            role = "User" if m["role"] == "user" else "Assistant"
            hist_str += f"{role}: {m['content']}\n"
        hist_str += "\n"

    docs = f" from {doc_names}" if doc_names else ""
    outline_str = (f"Outline of the whole document (its headings, use it for questions "
                   f"about what the document covers):\n{outline}\n\n") if outline else ""
    lang = _language_hint(question)

    return f"""Answer the question using ONLY the document passages{docs} below.

Rules:
1. Use only facts that are written in the passages. Do not add anything from your own knowledge.
2. If the passages do not contain the answer, reply only with this sentence (in {lang}): "The provided documents do not contain this information."
3. Copy numbers, dates and names exactly as they are written in the passages.
4. For questions about the document as a whole (what it is about, what it covers, a summary), use the outline and the passages and mention all main topics.
5. Keep the answer short and direct.

{outline_str}{hist_str}Document passages:
{context}

Question:
{question}

Write the answer in {lang}, even though the passages may be in another language.

Answer:"""


_SCRIPTS = [
    ("\u0600", "\u06ff", "Arabic"), ("\u0400", "\u04ff", "Russian"),
    ("\u0590", "\u05ff", "Hebrew"), ("\u0900", "\u097f", "Hindi"),
    ("\u3040", "\u30ff", "Japanese"), ("\uac00", "\ud7af", "Korean"),
    ("\u4e00", "\u9fff", "Chinese"), ("\u0e00", "\u0e7f", "Thai"),
]


# Common words of Latin-script languages, for naming the reply language offline
_WORDS = {
    "German": "ich du der die das und ist nicht wie was wer wo wann warum mein meine ein eine mit für auf hoch viele kann darf gibt es bei",
    "French": "je tu le la les et est pas comment quoi qui où quand pourquoi mon ma mes un une avec pour sur combien peux",
    "Spanish": "yo el la los las y es no cómo qué quién dónde cuándo por mi mis un una con para cuántos cuántas puedo",
    "Italian": "io il lo la gli e è non come cosa chi dove quando perché mio mia un una con per quanti quante posso",
    "Portuguese": "eu o a os as e é não como que quem onde quando por meu minha um uma com para quantos quantas posso",
    "Dutch": "ik de het een en is niet hoe wat wie waar wanneer waarom mijn met voor op hoeveel kan mag",
    "Turkish": "ben bir ve bu ne nasıl kim nerede neden benim için kaç mı mi var",
    "English": "i you the a an and is are not how what who where when why my with for on many much can do does",
}


def _language_hint(question: str) -> str:
    """Name the reply language: by script, or by common words for Latin scripts."""
    for lo, hi, name in _SCRIPTS:
        if sum(lo <= ch <= hi for ch in question) >= 2:
            return name
    import re
    tokens = set(re.findall(r"[^\W\d_]+", question.lower()))
    scores = {lang: sum(t in set(words.split()) for t in tokens) for lang, words in _WORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0 and list(scores.values()).count(scores[best]) == 1:
        return best
    return "the same language as the question"


def _resolve_model(model: str = None) -> str:
    if model:
        return model
    chosen = _load_config().get("chosen_model")
    if chosen and chosen != "auto":
        return chosen
    from rag.model_catalog import best_installed
    from rag.ollama_client import list_models
    return best_installed(list_models())


def ask(question: str, model: str = None, top_k: int = 5, history: list = None,
        dynamic_index: dict = None, doc_names: str = None) -> dict:
    """Full pipeline without streaming (used by the eval scripts)."""
    from rag.ollama_client import generate

    model = _resolve_model(model)
    chunks, retrieval_time = retrieve(_search_query(question, history), top_k, dynamic_index)
    prompt = build_rag_prompt(question, chunks, history, doc_names,
                              (dynamic_index or {}).get("outline"))
    system = f"{SYSTEM_PROMPT} Answer in {_language_hint(question)}."

    start = time.time()
    result = generate(model, prompt, system=system, temperature=0.1, max_tokens=512)
    answer = ensure_language(result["text"], question, model)
    return {
        "answer": answer,
        "sources": [c["chunk_id"] for c in chunks],
        "retrieval_time": retrieval_time,
        "generation_time": round(time.time() - start, 3),
        "model": model,
        "chunks": chunks,
    }


def ask_stream(question: str, model: str = None, top_k: int = 5, history: list = None,
               dynamic_index: dict = None, doc_names: str = None):
    """Full pipeline, answer streamed word by word. Returns (stream, chunks, retrieval_time, start)."""
    from rag.ollama_client import generate_stream

    model = _resolve_model(model)
    chunks, retrieval_time = retrieve(_search_query(question, history), top_k, dynamic_index)
    prompt = build_rag_prompt(question, chunks, history, doc_names,
                              (dynamic_index or {}).get("outline"))
    system = f"{SYSTEM_PROMPT} Answer in {_language_hint(question)}."

    start = time.time()
    stream = generate_stream(model, prompt, system=system, temperature=0.1, max_tokens=512)
    return stream, chunks, retrieval_time, start


def ensure_language(answer: str, question: str, model: str = None) -> str:
    """
    Small models often answer in the language of the passages instead of the
    question. If the answer is clearly in another language than the question,
    translate it with the same local model. Returns the answer unchanged otherwise.
    """
    from rag.ollama_client import generate

    want = _language_hint(question)
    got = _language_hint(answer)
    unknown = "the same language as the question"
    if want == unknown or got == unknown or want == got or not answer.strip():
        return answer
    result = generate(
        _resolve_model(model),
        f"Translate the following text into {want}. Keep all numbers and names exactly. "
        f"Reply with the translation only.\n\nText:\n{answer.strip()}",
        system=f"You are a translator. You always write in {want}.",
        temperature=0.0, max_tokens=512,
    )
    return result["text"].strip() or answer


_LLM_ONLY_PROMPT = "Answer briefly (at most 100 words), in the same language as the question.\n\nQuestion:\n{q}\n\nAnswer:"


def ask_llm_only(question: str, model: str = None) -> dict:
    """The model alone, without documents (the "Use RAG Sources" switch turned off)."""
    from rag.ollama_client import generate
    model = _resolve_model(model)
    start = time.time()
    result = generate(model, _LLM_ONLY_PROMPT.format(q=question), temperature=0.3, max_tokens=200)
    return {
        "answer": result["text"],
        "sources": [],
        "retrieval_time": 0,
        "generation_time": round(time.time() - start, 3),
        "model": model,
    }


def ask_llm_only_stream(question: str, model: str = None):
    from rag.ollama_client import generate_stream
    model = _resolve_model(model)
    start = time.time()
    stream = generate_stream(model, _LLM_ONLY_PROMPT.format(q=question), temperature=0.3, max_tokens=200)
    return stream, [], 0, start
