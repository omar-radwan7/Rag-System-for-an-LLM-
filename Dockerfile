# App image. Use with docker-compose.yml, which also starts Ollama.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# CPU build of torch first, so pip does not pull the CUDA packages
RUN pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rag ./rag
COPY ui ./ui
COPY config.json .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.port=8501", \
     "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
