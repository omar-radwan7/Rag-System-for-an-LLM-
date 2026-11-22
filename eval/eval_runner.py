#!/usr/bin/env python3
"""
eval_runner.py — Compare LLM-only vs RAG on your own documents.

Usage:
    python eval/eval_runner.py my_docs.pdf [more.pdf ...] --gold my_questions.json

The gold file is a list of {"question_id", "question", "expected_answer_short"}.
See eval/gold_questions.example.json. Results go to results/eval_results.csv.
"""

import json
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.rag_pipeline import ask, ask_llm_only
from rag.pdf_indexer import build_memory_index
from rag.ollama_client import generate

EVAL_DIR = os.path.dirname(__file__)
GOLD_PATH = os.path.join(EVAL_DIR, "gold_questions.example.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold(path=GOLD_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_eval_score(expected: str, actual: str) -> dict:
    """
    Heuristic scoring of answer quality by comparing with expected answer.
    """
    if not actual or actual.startswith("ERROR"):
        return {"accuracy_score": 1, "faithfulness_score": 1, "hallucination_flag": True}

    expected_words = set(expected.split())
    actual_words = set(actual.split())
    overlap = expected_words & actual_words
    overlap_ratio = len(overlap) / max(len(expected_words), 1)

    # Accuracy: how much of the expected answer is covered
    if overlap_ratio > 0.5:
        accuracy = 5
    elif overlap_ratio > 0.3:
        accuracy = 4
    elif overlap_ratio > 0.15:
        accuracy = 3
    elif overlap_ratio > 0.05:
        accuracy = 2
    else:
        accuracy = 1

    # Faithfulness: shorter, focused answers score higher
    word_count = len(actual.split())
    if word_count < 200 and overlap_ratio > 0.3:
        faithfulness = 5
    elif word_count < 300:
        faithfulness = 4
    elif word_count < 500:
        faithfulness = 3
    else:
        faithfulness = 2

    # Hallucination: flag if answer is very long but low overlap
    hallucination = word_count > 200 and overlap_ratio < 0.1

    return {
        "accuracy_score": accuracy,
        "faithfulness_score": faithfulness,
        "hallucination_flag": hallucination,
    }


def run_evaluation(pdf_paths, gold_path=GOLD_PATH):
    cfg = load_config()
    from rag.rag_pipeline import _resolve_model
    model = _resolve_model()
    gold = load_gold(gold_path)
    doc_index = build_memory_index([(os.path.basename(p), open(p, "rb").read()) for p in pdf_paths])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "eval_results.csv")
    fieldnames = [
        "question_id", "mode", "question", "expected_answer",
        "answer", "accuracy_score", "faithfulness_score",
        "hallucination_flag", "latency",
    ]

    rows = []
    llm_scores = []
    rag_scores = []

    print(f"🔬 Running evaluation with model: {model}")
    print(f"   Questions: {len(gold)}")
    print()

    for q in gold:
        qid = q["question_id"]
        question = q["question"]
        expected = q["expected_answer_short"]

        # ─── LLM Only ───
        print(f"  Q{qid} [LLM]...", end=" ", flush=True)
        try:
            llm_result = ask_llm_only(question, model=model)
            llm_answer = llm_result["answer"]
            llm_latency = llm_result["generation_time"]
        except Exception as e:
            llm_answer = f"ERROR: {e}"
            llm_latency = -1

        llm_eval = auto_eval_score(expected, llm_answer)
        llm_scores.append(llm_eval)
        rows.append({
            "question_id": qid, "mode": "LLM",
            "question": question, "expected_answer": expected,
            "answer": llm_answer,
            "accuracy_score": llm_eval["accuracy_score"],
            "faithfulness_score": llm_eval["faithfulness_score"],
            "hallucination_flag": llm_eval["hallucination_flag"],
            "latency": llm_latency,
        })
        print(f"done ({llm_latency}s)")

        # ─── RAG ───
        print(f"  Q{qid} [RAG]...", end=" ", flush=True)
        try:
            rag_result = ask(question, model=model, top_k=5, dynamic_index=doc_index,
                             doc_names=doc_index["doc_names"])
            rag_answer = rag_result["answer"]
            rag_latency = rag_result["retrieval_time"] + rag_result["generation_time"]
        except Exception as e:
            rag_answer = f"ERROR: {e}"
            rag_latency = -1

        rag_eval = auto_eval_score(expected, rag_answer)
        rag_scores.append(rag_eval)
        rows.append({
            "question_id": qid, "mode": "RAG",
            "question": question, "expected_answer": expected,
            "answer": rag_answer,
            "accuracy_score": rag_eval["accuracy_score"],
            "faithfulness_score": rag_eval["faithfulness_score"],
            "hallucination_flag": rag_eval["hallucination_flag"],
            "latency": rag_latency,
        })
        print(f"done ({rag_latency}s)")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Eval results saved to {csv_path}")

    # Write final report
    write_final_report(model, gold, llm_scores, rag_scores, rows)


def write_final_report(model, gold, llm_scores, rag_scores, rows):
    """Generate the final comparison report."""
    report_path = os.path.join(RESULTS_DIR, "final_report.md")

    def avg(scores, key):
        vals = [s[key] for s in scores if isinstance(s[key], (int, float))]
        return round(sum(vals) / max(len(vals), 1), 2)

    def count_flag(scores):
        return sum(1 for s in scores if s.get("hallucination_flag"))

    llm_rows = [r for r in rows if r["mode"] == "LLM" and r["latency"] > 0]
    rag_rows = [r for r in rows if r["mode"] == "RAG" and r["latency"] > 0]
    llm_avg_lat = round(sum(r["latency"] for r in llm_rows) / max(len(llm_rows), 1), 2)
    rag_avg_lat = round(sum(r["latency"] for r in rag_rows) / max(len(rag_rows), 1), 2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report: LLM only vs RAG\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Model Used:** `{model}`\n\n")
        f.write(f"**Number of Questions:** {len(gold)}\n\n")

        f.write("---\n\n")

        f.write("## 1. Survey Methodology\n\n")
        f.write("- Tested local models via Ollama (CPU mode only)\n")
        f.write("- Compared answers with and without the documents\n")
        f.write("- Evaluation: Automatic based on accuracy and faithfulness\n\n")

        f.write("## 2. Model Ranking\n\n")
        f.write("See `results/survey_summary.md` for detailed ranking.\n\n")

        f.write(f"## 3. Model Selection Justification\n\n")
        f.write(f"`{model}` was chosen based on the highest overall average score.\n\n")

        f.write("## 4. RAG System Architecture\n\n")
        f.write("```\n")
        f.write("User Question\n")
        f.write("    ↓\n")
        f.write("Embed Question (multilingual-e5-small)\n")
        f.write("    ↓\n")
        f.write("Retrieve top chunks from FAISS\n")
        f.write("    ↓\n")
        f.write("Build bilingual question with context\n")
        f.write("    ↓\n")
        f.write("Generate answer via Ollama\n")
        f.write("    ↓\n")
        f.write("Display answer + sources + latency\n")
        f.write("```\n\n")

        f.write("## 5. LLM vs RAG Comparison\n\n")
        f.write("| Metric | LLM Only | RAG |\n")
        f.write("|---------|---------|-----|\n")
        f.write(f"| Accuracy (1-5) | {avg(llm_scores, 'accuracy_score')} | {avg(rag_scores, 'accuracy_score')} |\n")
        f.write(f"| Faithfulness (1-5) | {avg(llm_scores, 'faithfulness_score')} | {avg(rag_scores, 'faithfulness_score')} |\n")
        f.write(f"| Hallucination Cases | {count_flag(llm_scores)}/{len(llm_scores)} | {count_flag(rag_scores)}/{len(rag_scores)} |\n")
        f.write(f"| Average Latency | {llm_avg_lat}s | {rag_avg_lat}s |\n\n")

        f.write("## 6. Observations\n\n")

        # Dynamic observations
        acc_diff = avg(rag_scores, 'accuracy_score') - avg(llm_scores, 'accuracy_score')
        if acc_diff > 0:
            f.write(f"- RAG system achieved higher accuracy by {acc_diff} points compared to the model alone.\n")
        elif acc_diff < 0:
            f.write(f"- The model alone achieved higher accuracy by {abs(acc_diff)} points — retrieval quality might be the reason.\n")
        else:
            f.write("- Accuracy is equal between both modes.\n")

        llm_hall = count_flag(llm_scores)
        rag_hall = count_flag(rag_scores)
        if rag_hall < llm_hall:
            f.write(f"- RAG reduced hallucination cases ({rag_hall} vs {llm_hall}).\n")

        f.write("- All models run in CPU mode — performance will improve significantly with GPU.\n")
        f.write("- Automatic evaluation is approximate — human evaluation would be more accurate.\n\n")

        f.write("## 7. Limitations\n\n")
        f.write("- All models run on CPU only (noticeable slowness).\n")
        f.write("- Evaluation is heuristic, not human.\n")
        f.write("- Model sizes are limited (3B-7B) due to space constraints.\n")
        f.write("- Small embedding model — a larger model might improve retrieval.\n\n")

        f.write("## 8. Conclusion\n\n")
        f.write(f"Evaluated the RAG system locally with the `{model}` model. ")
        f.write("The system includes an interactive chat interface, semantic retrieval from text, and a comprehensive comparison between ")
        f.write("the model's performance alone and its performance with the RAG system.\n")

    print(f"✅ Final report saved to {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-only vs RAG on your documents")
    parser.add_argument("pdfs", nargs="+", help="PDF documents to index")
    parser.add_argument("--gold", default=GOLD_PATH, help="questions with expected answers (JSON)")
    args = parser.parse_args()
    run_evaluation(args.pdfs, args.gold)
