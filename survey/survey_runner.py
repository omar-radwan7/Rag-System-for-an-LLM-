#!/usr/bin/env python3
"""
survey_runner.py — Run Arabic benchmark prompts against all Ollama models,
collect outputs, latencies, and write results to CSV + summary Markdown.
"""

import json
import csv
import os
import sys
import time

# Add project root so we can import rag.ollama_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.ollama_client import generate, list_models

SURVEY_PROMPTS = os.path.join(os.path.dirname(__file__), "survey_prompts.json")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_prompts():
    with open(SURVEY_PROMPTS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def run_survey():
    cfg = load_config()
    models = cfg["models"]
    prompts = load_prompts()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "survey_results.csv")

    fieldnames = [
        "model_name", "prompt_id", "category", "prompt_text",
        "output_text", "latency_seconds",
        "fluency", "coherence", "correctness",
        "instruction_following", "hallucination_control",
    ]

    rows = []
    model_scores = {m: [] for m in models}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Testing model: {model}")
        print(f"{'='*60}")
        for i, p in enumerate(prompts):
            pid = p["id"]
            cat = p["category"]
            prompt_text = p["prompt"]
            print(f"  [{i+1}/{len(prompts)}] {pid} ({cat})...", end=" ", flush=True)

            try:
                result = generate(model, prompt_text, temperature=0.7, max_tokens=1024)
                output = result["text"]
                latency = result["latency_seconds"]
            except Exception as e:
                output = f"ERROR: {e}"
                latency = -1

            # --- Automated heuristic scoring (1-5) ---
            scores = auto_score(prompt_text, output, cat)

            row = {
                "model_name": model,
                "prompt_id": pid,
                "category": cat,
                "prompt_text": prompt_text,
                "output_text": output,
                "latency_seconds": latency,
                **scores,
            }
            rows.append(row)
            model_scores[model].append(scores)
            print(f"done ({latency}s)")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Results saved to {csv_path}")

    # Generate summary and pick best model
    best_model = write_summary(models, model_scores, rows)
    cfg["chosen_model"] = best_model
    save_config(cfg)
    print(f"✅ Chosen model: {best_model}")


def auto_score(prompt: str, output: str, category: str) -> dict:
    """
    Simple heuristic auto-scoring (1-5) based on output characteristics.
    This gives a reasonable baseline; you can adjust scores manually later.
    """
    scores = {
        "fluency": 3,
        "coherence": 3,
        "correctness": 3,
        "instruction_following": 3,
        "hallucination_control": 3,
    }

    if not output or output.startswith("ERROR"):
        return {k: 1 for k in scores}

    text = output.strip()
    word_count = len(text.split())

    # Fluency: based on length and Arabic character ratio
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    total_chars = max(len(text), 1)
    arabic_ratio = arabic_chars / total_chars

    if arabic_ratio > 0.5 and word_count > 15:
        scores["fluency"] = 5
    elif arabic_ratio > 0.3 and word_count > 8:
        scores["fluency"] = 4
    elif arabic_ratio > 0.15:
        scores["fluency"] = 3
    elif arabic_ratio > 0.05:
        scores["fluency"] = 2
    else:
        scores["fluency"] = 1

    # Coherence: based on sentence structure
    sentences = [s.strip() for s in text.replace(".", "。").replace("。", ".").split(".") if s.strip()]
    if len(sentences) >= 3 and word_count > 20:
        scores["coherence"] = 5
    elif len(sentences) >= 2 and word_count > 10:
        scores["coherence"] = 4
    elif word_count > 5:
        scores["coherence"] = 3
    else:
        scores["coherence"] = 2

    # Instruction following: did it respond in Arabic when asked in Arabic?
    if arabic_ratio > 0.4:
        scores["instruction_following"] = 5
    elif arabic_ratio > 0.2:
        scores["instruction_following"] = 4
    elif arabic_ratio > 0.1:
        scores["instruction_following"] = 3
    else:
        scores["instruction_following"] = 2

    # Correctness: partially inferred from length & structure
    if word_count > 30 and arabic_ratio > 0.4:
        scores["correctness"] = 4
    elif word_count > 15:
        scores["correctness"] = 3
    else:
        scores["correctness"] = 2

    # Hallucination control: shorter but relevant is better than overly long
    if word_count < 300 and word_count > 10:
        scores["hallucination_control"] = 4
    elif word_count >= 300:
        scores["hallucination_control"] = 3
    else:
        scores["hallucination_control"] = 3

    return scores


def write_summary(models, model_scores, rows):
    """Write survey_summary.md and return the best model name."""
    summary_path = os.path.join(RESULTS_DIR, "survey_summary.md")

    metrics = ["fluency", "coherence", "correctness",
               "instruction_following", "hallucination_control"]

    model_avgs = {}
    for model in models:
        if not model_scores[model]:
            model_avgs[model] = {m: 0 for m in metrics}
            model_avgs[model]["overall"] = 0
            continue
        avgs = {}
        for m in metrics:
            vals = [s[m] for s in model_scores[model]]
            avgs[m] = round(sum(vals) / len(vals), 2)
        avgs["overall"] = round(sum(avgs.values()) / len(metrics), 2)
        model_avgs[model] = avgs

    # Compute average latency per model
    latencies = {}
    for model in models:
        model_rows = [r for r in rows if r["model_name"] == model and r["latency_seconds"] > 0]
        if model_rows:
            latencies[model] = round(sum(r["latency_seconds"] for r in model_rows) / len(model_rows), 2)
        else:
            latencies[model] = -1

    best_model = max(models, key=lambda m: model_avgs[m].get("overall", 0))

    # Category breakdown
    categories = sorted(set(r["category"] for r in rows))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Arabic LLM Survey — Summary Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Models tested:** {', '.join(models)}\n\n")
        f.write(f"**Number of prompts:** {len(rows) // len(models)}\n\n")

        f.write("## Overall Scores\n\n")
        f.write("| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control | **Overall** | Avg Latency (s) |\n")
        f.write("|-------|---------|-----------|-------------|-------------------|-----------------|-------------|------------------|\n")
        for model in models:
            a = model_avgs[model]
            f.write(f"| {model} | {a['fluency']} | {a['coherence']} | {a['correctness']} | "
                    f"{a['instruction_following']} | {a['hallucination_control']} | "
                    f"**{a['overall']}** | {latencies[model]} |\n")

        f.write(f"\n## 🏆 Best Model: `{best_model}`\n\n")
        f.write(f"Selected based on highest overall score ({model_avgs[best_model]['overall']}/5.0).\n\n")

        # Per-category breakdown
        f.write("## Per-Category Breakdown\n\n")
        for cat in categories:
            f.write(f"### {cat}\n\n")
            f.write("| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |\n")
            f.write("|-------|---------|-----------|-------------|-------------------|------------------|\n")
            for model in models:
                cat_scores = [s for s, r in zip(model_scores[model],
                              [row for row in rows if row["model_name"] == model])
                              if r["category"] == cat]
                if cat_scores:
                    avgs = {m: round(sum(s[m] for s in cat_scores) / len(cat_scores), 2) for m in metrics}
                    f.write(f"| {model} | {avgs['fluency']} | {avgs['coherence']} | {avgs['correctness']} | "
                            f"{avgs['instruction_following']} | {avgs['hallucination_control']} |\n")
            f.write("\n")

        f.write("## Methodology\n\n")
        f.write("- All models run locally via Ollama (CPU-only mode)\n")
        f.write("- Temperature: 0.7, Max tokens: 1024\n")
        f.write("- Scoring: automated heuristic (Arabic ratio, length, structure)\n")
        f.write("- Categories: MSA Writing, Summarization, Reading Comprehension, Logical Reasoning, Dialect Test\n")

    print(f"✅ Summary saved to {summary_path}")
    return best_model


if __name__ == "__main__":
    run_survey()
