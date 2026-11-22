"""
Ollama API client — lightweight wrapper around the Ollama REST API.
No external SDK needed; uses `requests` only.
"""

import os
import requests
import time
import json

# Set OLLAMA_URL to use an Ollama server elsewhere (docker-compose sets http://ollama:11434)
OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

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
            "repeat_penalty": 1.0,
            "top_p": 0.9,
            "top_k": 40,
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
            "repeat_penalty": 1.0,
            "top_p": 0.9,
            "top_k": 40,
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
    """
    Local model tags. Cloud models (Ollama "remote_host" models such as
    "...:cloud") are left out, because they would send the documents to a server.
    """
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    return [
        m["name"] for m in resp.json().get("models", [])
        if not m.get("remote_host") and not m["name"].endswith("cloud")
    ]


def pull_model(name: str):
    """
    Download a model through Ollama. Yields (status_text, fraction_done or None).
    Raises RuntimeError if Ollama reports an error.
    """
    with requests.post(f"{OLLAMA_BASE}/api/pull", json={"model": name, "stream": True},
                       stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            total, done = msg.get("total"), msg.get("completed")
            fraction = done / total if total and done is not None else None
            yield msg.get("status", ""), fraction
