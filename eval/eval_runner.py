#!/usr/bin/env python3
"""
eval_runner.py — Compare LLM-only vs RAG on the gold questions.
Generates eval_results.csv and contributes to final_report.md.
"""

import json
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.rag_pipeline import ask, ask_llm_only
from rag.ollama_client import generate

EVAL_DIR = os.path.dirname(__file__)
GOLD_PATH = os.path.join(EVAL_DIR, "gold_questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
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


def run_evaluation():
    cfg = load_config()
    model = cfg.get("chosen_model") or cfg["models"][0]
    gold = load_gold()
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
            rag_result = ask(question, model=model, top_k=5)
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
        f.write("# تقرير التقييم النهائي — نظام RAG لكتاب أنتيخريستوس\n\n")
        f.write(f"**التاريخ:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**النموذج المستخدم:** `{model}`\n\n")
        f.write(f"**عدد الأسئلة:** {len(gold)}\n\n")

        f.write("---\n\n")

        f.write("## 1. منهجية الاستبيان\n\n")
        f.write("- تم اختبار 3 نماذج محلية عبر Ollama (وضع CPU فقط)\n")
        f.write("- 30 سؤالاً عربياً في 5 فئات: كتابة فصحى، تلخيص، فهم قرائي، تفكير منطقي، لهجات\n")
        f.write("- التقييم: تلقائي بناءً على نسبة النص العربي، الطول، التركيب\n\n")

        f.write("## 2. ترتيب النماذج\n\n")
        f.write("راجع `results/survey_summary.md` للترتيب التفصيلي.\n\n")

        f.write(f"## 3. تبرير اختيار النموذج\n\n")
        f.write(f"تم اختيار `{model}` بناءً على أعلى متوسط درجات شامل في الاستبيان العربي.\n\n")

        f.write("## 4. هندسة نظام RAG\n\n")
        f.write("```\n")
        f.write("سؤال المستخدم\n")
        f.write("    ↓\n")
        f.write("تضمين السؤال (multilingual-e5-small)\n")
        f.write("    ↓\n")
        f.write("استرجاع أقرب 5 مقاطع من FAISS\n")
        f.write("    ↓\n")
        f.write("بناء سؤال عربي مع السياق\n")
        f.write("    ↓\n")
        f.write("توليد الإجابة عبر Ollama\n")
        f.write("    ↓\n")
        f.write("عرض الإجابة + المصادر + زمن الاستجابة\n")
        f.write("```\n\n")

        f.write("## 5. مقارنة LLM مقابل RAG\n\n")
        f.write("| المقياس | LLM فقط | RAG |\n")
        f.write("|---------|---------|-----|\n")
        f.write(f"| الدقة (1-5) | {avg(llm_scores, 'accuracy_score')} | {avg(rag_scores, 'accuracy_score')} |\n")
        f.write(f"| الأمانة (1-5) | {avg(llm_scores, 'faithfulness_score')} | {avg(rag_scores, 'faithfulness_score')} |\n")
        f.write(f"| حالات الهلوسة | {count_flag(llm_scores)}/{len(llm_scores)} | {count_flag(rag_scores)}/{len(rag_scores)} |\n")
        f.write(f"| متوسط زمن الاستجابة | {llm_avg_lat}ث | {rag_avg_lat}ث |\n\n")

        f.write("## 6. الملاحظات\n\n")

        # Dynamic observations
        acc_diff = avg(rag_scores, 'accuracy_score') - avg(llm_scores, 'accuracy_score')
        if acc_diff > 0:
            f.write(f"- نظام RAG حقق دقة أعلى بفارق {acc_diff} نقطة مقارنة بالنموذج وحده\n")
        elif acc_diff < 0:
            f.write(f"- النموذج وحده حقق دقة أعلى بفارق {abs(acc_diff)} نقطة — قد يكون السبب جودة الاسترجاع\n")
        else:
            f.write("- الدقة متساوية بين الوضعين\n")

        llm_hall = count_flag(llm_scores)
        rag_hall = count_flag(rag_scores)
        if rag_hall < llm_hall:
            f.write(f"- RAG قلّل من حالات الهلوسة ({rag_hall} مقابل {llm_hall})\n")

        f.write("- جميع النماذج تعمل في وضع CPU — الأداء سيتحسن كثيراً مع GPU\n")
        f.write("- التقييم التلقائي تقريبي — التقييم البشري سيكون أكثر دقة\n\n")

        f.write("## 7. القيود\n\n")
        f.write("- تعمل جميع النماذج على CPU فقط (بطء ملحوظ)\n")
        f.write("- التقييم هيوريستيكي وليس بشرياً\n")
        f.write("- حجم النماذج محدود (3B-7B) بسبب قيود المساحة\n")
        f.write("- نموذج التضمين صغير — نموذج أكبر قد يحسن الاسترجاع\n\n")

        f.write("## 8. الخلاصة\n\n")
        f.write(f"تم بنجاح بناء نظام RAG كامل لكتاب أنتيخريستوس باستخدام نموذج `{model}` محلياً. ")
        f.write("النظام يشمل واجهة محادثة تفاعلية، استرجاع دلالي من النص، ومقارنة شاملة بين ")
        f.write("أداء النموذج بمفرده وأدائه مع نظام RAG.\n")

    print(f"✅ Final report saved to {report_path}")


if __name__ == "__main__":
    run_evaluation()
