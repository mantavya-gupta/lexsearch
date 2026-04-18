"""
vector_store.py — Qdrant vector store for LexSearch
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import uuid

COLLECTION_NAME = "lexsearch_legal_docs"
VECTOR_SIZE = 384


def get_client(host="localhost", port=6333):
    try:
        client = QdrantClient(host=host, port=port)
        client.get_collections()
        print(f"[vector_store] Connected to Qdrant at {host}:{port}")
        return client
    except Exception:
        print("[vector_store] No Qdrant server found. Using in-memory mode.")
        return QdrantClient(":memory:")


def create_collection(client, recreate=False):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"[vector_store] Deleted existing collection.")
        else:
            print(f"[vector_store] Collection already exists.")
            return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[vector_store] Collection '{COLLECTION_NAME}' created.")


def upsert_chunks(client, embedded_chunks):
    if not embedded_chunks:
        return
    points = []
    for ec in embedded_chunks:
        chunk = ec.chunk
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_type": chunk.doc_type,
            "filename": chunk.filename,
            "title": chunk.title,
            "text": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "clause_type": chunk.clause_type or "unclassified",
            "section_header": chunk.section_header or "",
            "token_estimate": chunk.token_estimate,
            "chunk_index": chunk.chunk_index,
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=ec.embedding,
            payload=payload,
        ))
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"[vector_store] Upserted {min(i+batch_size, len(points))}/{len(points)} chunks")
    print(f"[vector_store] Done. {len(points)} chunks stored.")


def dense_search(client, query_embedding, top_k=20, filter_clause_type=None):
    search_filter = None
    if filter_clause_type:
        search_filter = Filter(
            must=[FieldCondition(
                key="clause_type",
                match=MatchValue(value=filter_clause_type)
            )]
        )
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    ).points

    return [
        {
            "text": r.payload["text"],
            "score": round(r.score, 4),
            "filename": r.payload["filename"],
            "page_start": r.payload["page_start"],
            "clause_type": r.payload["clause_type"],
            "chunk_id": r.payload["chunk_id"],
            "doc_id": r.payload["doc_id"],
            "title": r.payload["title"],
        }
        for r in results
    ]


def get_collection_info(client):
    info = client.get_collection(COLLECTION_NAME)
    return {
        "total_chunks": info.points_count,
        "collection": COLLECTION_NAME,
        "vector_size": VECTOR_SIZE,
        "status": info.status,
    }
