"""
Ollama API client — lightweight wrapper around the Ollama REST API.
No external SDK needed; uses `requests` only.
"""

import requests
import time
import json

OLLAMA_BASE = "http://localhost:11434"

def generate(model: str, prompt: str, system: str = "", temperature: float = 0.7,
             max_tokens: int = 150) -> dict:
    """
    Call Ollama /api/generate and return the full response text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    start = time.time()
    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=600)
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data.get("response", ""),
        "latency_seconds": round(latency, 2),
        "total_duration_ns": data.get("total_duration", 0),
        "eval_count": data.get("eval_count", 0),
    }

def generate_stream(model: str, prompt: str, system: str = "", temperature: float = 0.7,
                    max_tokens: int = 150):
    """
    Call Ollama /api/generate with stream=True. Yields text chunks.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    with requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")

def list_models() -> list:
    """Return the list of locally available model tags."""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
