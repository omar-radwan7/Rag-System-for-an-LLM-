#!/usr/bin/env python3
"""
keyword_search.py — BM25 keyword search over the chunks, fused with the vector search.

Vector search can miss passages that share the exact names, numbers and terms of
the question. BM25 catches those, and Reciprocal Rank Fusion combines both rankings
without having to calibrate their scores against each other. The tokenizer handles
English and Arabic (light Arabic normalization and prefix stripping).
"""

import math
import re
from collections import Counter, defaultdict

_STOPWORDS = set(
    "في من على إلى الى عن أن ان ما لا هل كان كانت التي الذي هو هي ثم قد لم لن مع كل "
    "بعد قبل ذلك هذا هذه ماذا كيف لماذا متى اين أين كم".split()
)
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")


def _normalize(text: str) -> str:
    text = re.sub("[ً-ْـ]", "", text)  # diacritics and tatweel
    text = re.sub("[أإآ]", "ا", text)
    return text.replace("ة", "ه").replace("ى", "ي")


def tokenize(text: str) -> list:
    tokens = []
    for word in re.findall(r"[ء-ي]+|[a-zA-Z0-9]+", _normalize(text).lower()):
        # Light stemming: drop the article and attached و so خالد / وخالد / الخالد match
        for prefix in _PREFIXES:
            if word.startswith(prefix) and len(word) - len(prefix) >= 3:
                word = word[len(prefix):]
                break
        if word.startswith("و") and len(word) > 4:
            word = word[1:]
        if len(word) > 1 and word not in _STOPWORDS:
            tokens.append(word)
    return tokens


class BM25:
    def __init__(self, texts: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        self.postings = defaultdict(list)  # term -> [(doc index, term frequency)]
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.n_docs = len(docs)

    def rank(self, query: str, limit: int = 100) -> list:
        """Document indices ordered by BM25 score (only documents sharing a term)."""
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores, key=scores.get, reverse=True)[:limit]


def reciprocal_rank_fusion(rankings: list, k: int = 60) -> dict:
    """Combine several rankings of document indices into one score per index."""
    fused = defaultdict(float)
    for ranking in rankings:
        for position, doc in enumerate(ranking):
            fused[doc] += 1 / (k + position)
    return fused
