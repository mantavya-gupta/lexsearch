# LexSearch — AI-powered Legal Document Intelligence

Ask natural language questions about legal contracts and get grounded answers with source citations.

## Architecture

PDF Upload → PyMuPDF Parser → Smart Legal Chunker → Sentence Transformers → Qdrant Vector Store + BM25 → Hybrid Search → Query Expansion (HyDE) → LLaMA3 via Groq → FastAPI REST API

## Key Features

- Hybrid dense + BM25 search fusion
- Query expansion with HyDE (3 variants per query)
- Auto clause detection: indemnity, liability, termination, payment, confidentiality
- Grounded answers with page-level source citations
- Conversation memory per session
- Professional web UI

## Quick Start

    git clone https://github.com/mantavya-gupta/lexsearch
    cd lexsearch/backend
    pip install -r requirements.txt
    export GROQ_API_KEY=your-key-here
    uvicorn app.main:app --reload --port 8000

## API Endpoints

- GET  /api/v1/health
- POST /api/v1/upload
- POST /api/v1/query
- GET  /api/v1/documents

## Tech Stack

Python 3.11, FastAPI, Qdrant, sentence-transformers, rank-bm25, Groq LLaMA3, PyMuPDF, Docker

## Author

Built by Mantavya Gupta — production-grade RAG system for legal document intelligence.
