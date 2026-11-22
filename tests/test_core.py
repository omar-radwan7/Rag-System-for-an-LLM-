"""Fast tests without models or Ollama: python -m pytest"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.pdf_indexer import chunk_text, find_headings, normalize_text
from rag.rag_pipeline import _language_hint
from rag.keyword_search import BM25
from rag.model_catalog import CATALOG, check


def test_chunks_overlap_and_size():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, 500, 150)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]


def test_normalize_removes_page_numbers_and_joins_lines():
    assert normalize_text("Line one\ncontinues here\n12\n") == "Line one continues here"


def test_headings():
    raw = "Handbook\n1. Working hours\nCore hours are from 10 to 15 and this line is long.\n"
    assert find_headings(raw) == ["Handbook", "1. Working hours"]


def test_language_hint():
    assert _language_hint("كم يوما يمكنني العمل؟") == "Arabic"
    assert _language_hint("Wie hoch ist mein Budget?") == "German"
    assert _language_hint("How many days do I get?") == "English"


def test_bm25_ranks_matching_text_first():
    bm25 = BM25(["the vacation policy", "laptop and vpn rules"])
    assert bm25.rank("vpn")[0] == 1


def test_model_check():
    small = CATALOG[0]
    assert check(small, set(), ram_gb=16, disk_gb=50)[0] == "ok"
    assert check(small, {small["name"]})[0] == "installed"
    assert check(CATALOG[-1], set(), ram_gb=8, disk_gb=50)[0] == "ram"
    assert check(small, set(), ram_gb=16, disk_gb=0.5)[0] == "disk"
