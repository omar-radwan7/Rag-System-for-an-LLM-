#!/usr/bin/env python3
"""
models.py — Settings that depend on which embedding model config.json selects.

e5 models were trained with "query: " / "passage: " prefixes; other models (bge-m3)
expect plain text.
"""

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def query_text(model_name: str, text: str) -> str:
    return f"query: {text}" if _is_e5(model_name) else text


def passage_text(model_name: str, text: str) -> str:
    return f"passage: {text}" if _is_e5(model_name) else text
