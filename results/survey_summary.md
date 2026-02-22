# Arabic LLM Survey — Summary Report

**Date:** 2026-02-22 15:29

**Models tested:** qwen2.5:3b, llama3.2:3b, gemma2:2b

**Number of prompts:** 28

## Overall Scores

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control | **Overall** | Avg Latency (s) |
|-------|---------|-----------|-------------|-------------------|-----------------|-------------|------------------|
| qwen2.5:3b | 4.96 | 4.61 | 3.86 | 5.0 | 3.96 | **4.48** | 29.25 |
| llama3.2:3b | 4.5 | 3.89 | 3.18 | 4.82 | 3.79 | **4.04** | 24.85 |
| gemma2:2b | 4.18 | 4.43 | 3.43 | 4.5 | 3.93 | **4.09** | 19.0 |

## 🏆 Best Model: `qwen2.5:3b`

Selected based on highest overall score (4.48/5.0).

## Per-Category Breakdown

### Dialect Test

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |
|-------|---------|-----------|-------------|-------------------|------------------|
| qwen2.5:3b | 5.0 | 5.0 | 4.0 | 5.0 | 4.0 |
| llama3.2:3b | 3.6 | 3.8 | 2.8 | 4.4 | 3.6 |
| gemma2:2b | 4.0 | 5.0 | 3.6 | 4.0 | 4.0 |

### Logical Reasoning

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |
|-------|---------|-----------|-------------|-------------------|------------------|
| qwen2.5:3b | 5.0 | 5.0 | 4.0 | 5.0 | 4.0 |
| llama3.2:3b | 4.67 | 4.0 | 3.17 | 4.67 | 4.0 |
| gemma2:2b | 4.17 | 5.0 | 3.67 | 4.5 | 4.0 |

### MSA Writing

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |
|-------|---------|-----------|-------------|-------------------|------------------|
| qwen2.5:3b | 5.0 | 5.0 | 4.0 | 5.0 | 4.0 |
| llama3.2:3b | 5.0 | 4.67 | 3.83 | 5.0 | 3.83 |
| gemma2:2b | 4.33 | 5.0 | 3.83 | 4.5 | 4.0 |

### Reading Comprehension

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |
|-------|---------|-----------|-------------|-------------------|------------------|
| qwen2.5:3b | 4.8 | 4.0 | 3.6 | 5.0 | 3.8 |
| llama3.2:3b | 4.2 | 3.2 | 2.8 | 5.0 | 3.6 |
| gemma2:2b | 4.0 | 3.2 | 2.6 | 5.0 | 3.6 |

### Summarization

| Model | Fluency | Coherence | Correctness | Instr. Following | Halluc. Control |
|-------|---------|-----------|-------------|-------------------|------------------|
| qwen2.5:3b | 5.0 | 4.0 | 3.67 | 5.0 | 4.0 |
| llama3.2:3b | 4.83 | 3.67 | 3.17 | 5.0 | 3.83 |
| gemma2:2b | 4.33 | 3.83 | 3.33 | 4.5 | 4.0 |

## Methodology

- All models run locally via Ollama (CPU-only mode)
- Temperature: 0.7, Max tokens: 1024
- Scoring: automated heuristic (Arabic ratio, length, structure)
- Categories: MSA Writing, Summarization, Reading Comprehension, Logical Reasoning, Dialect Test
