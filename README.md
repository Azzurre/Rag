# Local RAG Assistant

A local Retrieval Augmented Generation project built with Python, ChromaDB, Hugging Face SentenceTransformers, and Ollama.

## What it does

This project allows a user to ask questions about local text documents. The system retrieves the most relevant document chunks using semantic search and sends them to a local LLM to generate grounded answers.

## Tech stack

- Python
- ChromaDB
- SentenceTransformers
- Ollama
- VS Code

## How it works

1. Documents are placed inside the `data/` folder.
2. `src/ingest.py` loads the documents.
3. The text is split into overlapping chunks.
4. Each chunk is converted into an embedding.
5. Chunks and embeddings are stored in ChromaDB.
6. `src/ask.py` embeds the user question.
7. ChromaDB retrieves relevant chunks.
8. Ollama generates an answer using the retrieved context.

## Setup

Create a virtual environment:

```bash
python -m venv .venv