#!/usr/bin/env bash
# One command setup and start for Linux and macOS:   ./run.sh
#
# 1. Creates a Python environment in .venv (first run only)
# 2. Installs the pinned packages (CPU build of torch on Linux)
# 3. Checks that Ollama is installed and running, starts it if needed
# 4. Downloads the best model this machine can run, if it is missing
# 5. Starts the app in the browser
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nError: %s\n' "$1" >&2; exit 1; }

# Python 3.10 or newer
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
        PY="$candidate"; break
    fi
done
[ -n "$PY" ] || fail "Python 3.10 to 3.13 is needed. Install it from https://www.python.org/downloads/"

if [ ! -x .venv/bin/python ]; then
    say "Creating the Python environment (.venv)"
    "$PY" -m venv .venv || fail "Could not create .venv. On Debian/Ubuntu run: sudo apt install python3-venv"
fi
VENV_PY=.venv/bin/python
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

if ! "$VENV_PY" -c "import streamlit, sentence_transformers, faiss" 2>/dev/null; then
    say "Installing packages (first run only, a few minutes)"
    "$VENV_PY" -m pip install -q --upgrade pip
    if [ "$(uname -s)" = "Linux" ]; then
        # CPU build: avoids about 3 GB of CUDA packages the app does not use
        "$VENV_PY" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
    fi
    "$VENV_PY" -m pip install -q -r requirements.txt
fi

# Ollama
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export OLLAMA_URL
if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    command -v ollama >/dev/null 2>&1 || fail "Ollama is not installed. Get it from https://ollama.com/download and run ./run.sh again."
    say "Starting Ollama"
    (ollama serve >/dev/null 2>&1 &)
    for _ in $(seq 1 30); do
        curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || fail "Ollama did not start. Start it yourself with: ollama serve"
fi

# chosen_model "auto" in config.json: the best model this machine can run
# (RAM, NVIDIA GPU memory, free disk). A fixed name in config.json is used as is.
MODEL="$("$VENV_PY" -m rag.model_catalog --recommend)"
if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL\""; then
    say "Downloading $MODEL, the best model for this machine (first run only)"
    curl -sf "$OLLAMA_URL/api/pull" -d "{\"model\": \"$MODEL\", \"stream\": false}" >/dev/null ||
        fail "Could not download $MODEL. Check your internet connection."
fi

say "Starting the app at http://localhost:${PORT:-8501}"
exec "$VENV_PY" -m streamlit run ui/app.py --server.port "${PORT:-8501}" --server.fileWatcherType none
