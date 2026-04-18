"""
main.py — FastAPI application entry point for LexSearch
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app.retrieval.vector_store import get_client, create_collection
from app.retrieval.hybrid_search import BM25Index
from app.api.routes import router

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="LexSearch API",
    description="AI-powered legal document intelligence system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (shared across requests) ──────────────────
qdrant_client = None
bm25_index = BM25Index()
embedded_store = []          # All embedded chunks in memory
embedding_model = None       # Loaded once on startup


@app.on_event("startup")
async def startup():
    global qdrant_client, embedding_model
    print("\n[LexSearch] Starting up...")

    # Connect to Qdrant
    qdrant_client = get_client()
    create_collection(qdrant_client, recreate=False)

    # Load embedding model once (avoids reloading per query)
    print("[LexSearch] Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[LexSearch] Ready!")


# ── Inject shared embedding model into rag_chain ──────────
import app.generation.rag_chain as rag_module

def _fast_embed(text):
    global embedding_model
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

rag_module.embed_query = _fast_embed

# ── Routes ─────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "LexSearch",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
