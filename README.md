# RAG System for Enterprise Documentation

## Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to transform static documents (PDFs) into interactive, context-aware knowledge bases. It explores the intersection of data privacy, local Large Language Model (LLM) inference, and multilingual information retrieval.

Developed as part of a research and development initiative at the Applied Innovation Center (AIC), the system addresses the need for organizations to interact with sensitive internal data without uploading it to external servers. This project serves as the foundation for a bachelor's thesis focused on privacy-preserving, locally deployed RAG architectures.

## Problem Statement

Modern Large Language Models face key challenges when applied to enterprise use cases:

1. **Knowledge Cutoffs and Hallucinations**: Standard LLMs are limited to their training data and often produce factually incorrect responses when asked about documents not included in their pre-training.
2. **Data Privacy and Sovereignty**: Cloud-based AI solutions require uploading sensitive documents to external servers, which is often prohibited by institutional security policies or regulations such as GDPR.
3. **Multilingual Support**: The system is designed to process and respond in multiple languages, since enterprise documentation is not always written in a single language.

## The Proposed Solution

This system implements a fully local RAG pipeline. By decoupling the knowledge base from the model's internal weights, the system ensures that:

- Information is retrieved directly from the provided source documents, reducing hallucinations regarding document content.
- All processing occurs on local hardware via Ollama, ensuring no data leaves the local machine.
- The system supports multilingual queries through a multilingual embedding model.

## Key Technical Features

### In-Memory Vector Indexing

When a user uploads a PDF, the system performs text extraction, sliding-window chunking, and vector embedding. These vectors are stored in a local FAISS (Facebook AI Similarity Search) index, allowing fast retrieval without external database dependencies.

### Multilingual Embeddings

The system uses a multilingual embedding model, allowing it to process and retrieve relevant context across different languages without requiring separate pipelines per language.

### Strict Context-Grounded Generation

The system uses a constrained prompt template that instructs the model to answer only using the retrieved document context. If the requested information is not present in the retrieved chunks, the model is instructed to state that the information is not available, rather than generating a speculative answer.

## Technical Architecture

The pipeline is structured into three layers:

1. **Ingestion Layer**: Extracts text from PDFs and applies sliding-window chunking to preserve context across chunk boundaries.
2. **Retrieval Layer**: Uses FAISS for similarity search over document embeddings.
3. **Generation Layer**: Uses the Ollama REST API to interface with local, quantized open-source models such as Qwen2.5 and Llama 3.

## Research Value

This project supports investigation into:

- **Efficiency vs. Accuracy**: How chunk size and overlap affect retrieval precision.
- **Local Inference Benchmarking**: Trade-offs between model quantization (e.g., 4-bit) and response quality on consumer-grade hardware.
- **Hallucination Mitigation**: Whether strict prompt constraints can reliably prevent a language model from generating unsupported answers.

## Project Structure

```
Rag-System-for-an-LLM/
├── rag/                 # Core RAG pipeline
│   ├── pdf_indexer.py   # PDF processing and FAISS indexing
│   ├── rag_pipeline.py  # Prompt construction and retrieval logic
│   └── ollama_client.py # Wrapper for the Ollama REST API
├── ui/                  # Frontend interface
│   ├── app.py            # Streamlit chat interface
│   └── chat_storage.json # Session management
├── eval/                 # Evaluation framework
│   ├── eval_runner.py    # Performance and accuracy benchmarking
│   └── gold_questions.json # Standardized test set
├── data/                  # Dataset management and preprocessing
├── results/               # Performance logs and benchmarking outputs
└── config.json            # System configuration
```

## Getting Started

### Prerequisites

- Ollama installed and running (https://ollama.ai)
- Python 3.9 or higher

### Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`

### Model Configuration

The system is optimized for the Qwen2.5:3B model, which offers a balance between response quality and performance on CPU-only hardware.

Pull the model:
```
ollama pull qwen2.5:3b
```

### Execution

Run the Streamlit interface:
```
streamlit run ui/app.py
```

## License

MIT License
