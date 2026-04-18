"""
routes.py — FastAPI routes for LexSearch
Endpoints: /health /upload /query /documents /clear-session
"""

import os
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import upsert_chunks, get_collection_info
from app.retrieval.hybrid_search import build_bm25_from_embedded
from app.generation.rag_chain import run_rag_query
from app.generation.memory import ConversationMemory
from .models import QueryRequest, QueryResponse, UploadResponse, HealthResponse, ClearResponse

router = APIRouter()

# Session memory store — maps session_id -> ConversationMemory
sessions: dict[str, ConversationMemory] = {}


def get_session(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory()
    return sessions[session_id]


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check API status and how many chunks are indexed."""
    from app.main import qdrant_client
    try:
        info = get_collection_info(qdrant_client)
        chunks = info["total_chunks"]
    except Exception:
        chunks = 0
    return HealthResponse(
        status="ok",
        chunks_indexed=chunks,
        model="llama-3.1-8b-instant (Groq) + all-MiniLM-L6-v2",
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF legal document.
    Parses, chunks, embeds, and indexes it into Qdrant automatically.
    """
    from app.main import qdrant_client, bm25_index, embedded_store

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Parse
        doc = parse_pdf(tmp_path)
        doc.filename = file.filename  # Use original filename

        # Chunk
        chunks = chunk_document(doc)

        # Embed
        embedded = embed_chunks(
            chunks,
            cache_path=Path("data/processed/.cache/embeddings.json")
        )

        # Store in Qdrant
        upsert_chunks(qdrant_client, embedded)

        # Update BM25 index
        embedded_store.extend(embedded)
        bm25_index.build([
            {
                "text": ec.chunk.text,
                "chunk_id": ec.chunk.chunk_id,
                "doc_id": ec.chunk.doc_id,
                "filename": ec.chunk.filename,
                "title": ec.chunk.title,
                "page_start": ec.chunk.page_start,
                "clause_type": ec.chunk.clause_type or "unclassified",
            }
            for ec in embedded_store
        ])

        return UploadResponse(
            filename=file.filename,
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            page_count=doc.page_count,
            chunks_created=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks from {file.filename}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about your uploaded legal documents.
    Returns a grounded answer with source citations.
    """
    from app.main import qdrant_client, bm25_index

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        memory = get_session(request.session_id)

        result = run_rag_query(
            question=request.question,
            qdrant_client=qdrant_client,
            bm25_index=bm25_index,
        )

        memory.add(request.question, result["answer"])

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                {
                    "filename": s.get("filename", "unknown"),
                    "page": s.get("page", 1),
                    "clause_type": s.get("clause_type", "general"),
                    "score": round(float(s.get("score", 0)), 4),
                    "preview": s.get("preview", ""),
                }
                for s in result["sources"]
            ],
            chunks_retrieved=result["chunks_retrieved"],
            chunks_used=result["chunks_used"],
            session_id=request.session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """List all indexed documents."""
    from app.main import embedded_store
    docs = {}
    for ec in embedded_store:
        doc_id = ec.chunk.doc_id
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": ec.chunk.filename,
                "doc_type": ec.chunk.doc_type,
                "title": ec.chunk.title,
                "chunks": 0,
            }
        docs[doc_id]["chunks"] += 1
    return {"documents": list(docs.values()), "total": len(docs)}


@router.post("/clear-session", response_model=ClearResponse)
async def clear_session(session_id: str = "default"):
    """Clear conversation memory for a session."""
    if session_id in sessions:
        sessions[session_id].clear()
    return ClearResponse(
        message=f"Session cleared.",
        session_id=session_id,
    )
