"""
hybrid_search.py — Hybrid dense + BM25 search for LexSearch
Combines semantic vector search with keyword-based BM25 for better retrieval.
"""

import os
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from .vector_store import dense_search

BM25_INDEX_PATH = Path("data/processed/bm25_index.json")


class BM25Index:
    """
    Lightweight BM25 keyword index over all stored chunks.
    Built once from chunks, saved to disk, loaded on startup.
    """

    def __init__(self):
        self.chunks = []       # list of chunk dicts
        self.corpus = []       # tokenized texts
        self.bm25 = None

    def build(self, chunks):
        """Build BM25 index from a list of chunk dicts."""
        self.chunks = chunks
        self.corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[hybrid_search] BM25 index built: {len(chunks)} chunks")

    def save(self, path=BM25_INDEX_PATH):
        """Save index data to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"chunks": self.chunks}, f)
        print(f"[hybrid_search] BM25 index saved to {path}")

    def load(self, path=BM25_INDEX_PATH):
        """Load and rebuild index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}. Run build() first.")
        with open(path) as f:
            data = json.load(f)
        self.build(data["chunks"])
        print(f"[hybrid_search] BM25 index loaded from {path}")

    def search(self, query, top_k=20):
        """Return top_k chunks by BM25 score."""
        if not self.bm25:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self.chunks[i], "bm25_score": round(float(scores[i]), 4)}
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text):
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()


def build_bm25_from_embedded(embedded_chunks, save=True):
    """
    Build and optionally save a BM25 index from EmbeddedChunk objects.
    Call this right after upserting to Qdrant.
    """
    chunk_dicts = [
        {
            "text": ec.chunk.text,
            "chunk_id": ec.chunk.chunk_id,
            "doc_id": ec.chunk.doc_id,
            "filename": ec.chunk.filename,
            "title": ec.chunk.title,
            "page_start": ec.chunk.page_start,
            "clause_type": ec.chunk.clause_type or "unclassified",
        }
        for ec in embedded_chunks
    ]
    index = BM25Index()
    index.build(chunk_dicts)
    if save:
        index.save()
    return index


def hybrid_search(
    query,
    query_embedding,
    qdrant_client,
    bm25_index,
    top_k=20,
    dense_weight=0.7,
    bm25_weight=0.3,
):
    """
    Hybrid search: combine dense + BM25 results using weighted score fusion.

    Args:
        query:           The user's natural language question
        query_embedding: Dense embedding of the query
        qdrant_client:   Connected Qdrant client
        bm25_index:      Built BM25Index object
        top_k:           Number of results to return before re-ranking
        dense_weight:    Weight for dense scores (default 0.7)
        bm25_weight:     Weight for BM25 scores (default 0.3)

    Returns:
        List of result dicts sorted by combined score, highest first.
    """
    # 1. Get dense results
    dense_results = dense_search(qdrant_client, query_embedding, top_k=top_k)

    # 2. Get BM25 results
    bm25_results = bm25_index.search(query, top_k=top_k)

    # 3. Normalize scores to [0, 1]
    def normalize(results, score_key):
        scores = [r[score_key] for r in results]
        max_s = max(scores) if scores else 1
        min_s = min(scores) if scores else 0
        rng = max_s - min_s or 1
        for r in results:
            r[f"{score_key}_norm"] = (r[score_key] - min_s) / rng
        return results

    dense_results = normalize(dense_results, "score")
    bm25_results = normalize(bm25_results, "bm25_score")

    # 4. Merge by chunk_id
    merged = {}
    for r in dense_results:
        cid = r["chunk_id"]
        merged[cid] = {**r, "combined_score": dense_weight * r["score_norm"]}

    for r in bm25_results:
        cid = r["chunk_id"]
        if cid in merged:
            merged[cid]["combined_score"] += bm25_weight * r["bm25_score_norm"]
        else:
            merged[cid] = {
                **r,
                "score": 0,
                "score_norm": 0,
                "combined_score": bm25_weight * r["bm25_score_norm"],
            }

    # 5. Sort by combined score
    sorted_results = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
    return sorted_results[:top_k]
