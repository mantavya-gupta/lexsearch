"""
embedder.py — Free local embeddings for LexSearch
Uses sentence-transformers (all-MiniLM-L6-v2) — runs on Mac M-series, no API key needed.
Model downloads once (~90MB), then works completely offline.
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384


@dataclass
class EmbeddedChunk:
    chunk: object
    embedding: list
    model: str
    dimensions: int


def _cache_key(text):
    return hashlib.md5(f"{MODEL_NAME}:{text}".encode()).hexdigest()


def _load_cache(cache_path):
    if Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache, cache_path):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def embed_chunks(chunks, model="local", cache_path=Path(".cache/embeddings.json")):
    """
    Embed chunks using sentence-transformers locally.
    First run downloads ~90MB model. All subsequent runs use cache.
    """
    if not chunks:
        return []

    from sentence_transformers import SentenceTransformer

    cache = _load_cache(cache_path)
    texts = [c.text for c in chunks]
    results = [None] * len(texts)
    to_embed = []

    for i, text in enumerate(texts):
        key = _cache_key(text)
        if key in cache:
            results[i] = cache[key]
        else:
            to_embed.append((i, text))

    if to_embed:
        print(f"[embedder] Loading sentence-transformers model (downloads once ~90MB)...")
        st_model = SentenceTransformer(MODEL_NAME)
        batch_texts = [text for _, text in to_embed]
        print(f"[embedder] Embedding {len(batch_texts)} chunks locally...")
        embeddings = st_model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()
        for j, (orig_idx, text) in enumerate(to_embed):
            results[orig_idx] = embeddings[j]
            cache[_cache_key(text)] = embeddings[j]
        _save_cache(cache, cache_path)
        print(f"[embedder] Done. {len(batch_texts)} chunks embedded.")
    else:
        print(f"[embedder] All {len(texts)} embeddings loaded from cache.")

    return [
        EmbeddedChunk(chunk=c, embedding=e, model=MODEL_NAME, dimensions=DIMENSIONS)
        for c, e in zip(chunks, results) if e is not None
    ]
