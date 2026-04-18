"""
pipeline.py — Phase 1 orchestration for LexSearch
Ties together parser → chunker → embedder in one clean pipeline.
Run directly as a script to ingest a folder of PDFs:

    python -m app.ingestion.pipeline --input data/raw --model openai

Or import and call run_ingestion_pipeline() from your FastAPI routes.
"""

import json
import argparse
import time
from pathlib import Path
from dataclasses import asdict

from .parser import parse_directory, ParsedDocument
from .chunker import chunk_documents, chunk_stats, Chunk
from .embedder import embed_chunks, EmbeddedChunk


# ── Output helpers ─────────────────────────────────────────────────────────────

def _save_chunks_json(chunks: list[Chunk], output_dir: Path) -> Path:
    """Save all chunks to processed/chunks.json for inspection / debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "chunks.json"

    serializable = []
    for c in chunks:
        d = asdict(c)
        serializable.append(d)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[pipeline] Chunks saved to {out_path}")
    return out_path


def _save_stats(stats: dict, embedded_count: int, output_dir: Path) -> None:
    """Save a pipeline run summary for your README / evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **stats,
        "embedded_chunks": embedded_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = output_dir / "ingestion_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pipeline] Summary saved to {out_path}")
    return summary


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingestion_pipeline(
    input_dir: str | Path,
    output_dir: str | Path = "data/processed",
    embed_model: str = "openai",
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    recursive: bool = False,
    skip_embedding: bool = False,
) -> dict:
    """
    Full Phase 1 ingestion pipeline: Parse → Chunk → Embed.

    Args:
        input_dir:            Directory containing source PDFs
        output_dir:           Where to write processed chunks and stats
        embed_model:          "openai" or "bge-m3"
        chunk_size_tokens:    Target chunk size in tokens (default 512)
        chunk_overlap_tokens: Token overlap between chunks (default 50)
        recursive:            Search subdirectories for PDFs
        skip_embedding:       If True, stop after chunking (useful for testing)

    Returns:
        Dict with pipeline statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    t_start = time.time()

    print("\n" + "="*60)
    print("LexSearch — Phase 1 Ingestion Pipeline")
    print("="*60)
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Model:   {embed_model}")
    print(f"  Chunks:  {chunk_size_tokens} tokens, {chunk_overlap_tokens} overlap")
    print("="*60 + "\n")

    # ── Step 1: Parse ──────────────────────────────────────────────────────────
    print("► Step 1/3: Parsing PDFs...")
    parsed_docs: list[ParsedDocument] = parse_directory(input_dir, recursive=recursive)

    if not parsed_docs:
        print("[pipeline] No documents parsed. Exiting.")
        return {"error": "No documents parsed"}

    # ── Step 2: Chunk ──────────────────────────────────────────────────────────
    print("\n► Step 2/3: Chunking documents...")
    chunks: list[Chunk] = chunk_documents(
        parsed_docs,
        chunk_size=chunk_size_tokens * 4,       # tokens → chars
        chunk_overlap=chunk_overlap_tokens * 4,
    )

    stats = chunk_stats(chunks)
    print(f"\n[pipeline] Chunk stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    _save_chunks_json(chunks, output_dir)

    if skip_embedding:
        print("\n[pipeline] --skip-embedding set. Stopping after chunking.")
        _save_stats(stats, 0, output_dir)
        return stats

    # ── Step 3: Embed ──────────────────────────────────────────────────────────
    print(f"\n► Step 3/3: Generating embeddings ({embed_model})...")
    embedded: list[EmbeddedChunk] = embed_chunks(
        chunks,
        model=embed_model,
        cache_path=output_dir / ".cache" / "embeddings.json",
    )

    elapsed = round(time.time() - t_start, 1)
    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete in {elapsed}s")
    print(f"  Documents parsed:  {len(parsed_docs)}")
    print(f"  Chunks created:    {len(chunks)}")
    print(f"  Chunks embedded:   {len(embedded)}")
    print(f"{'='*60}\n")

    summary = _save_stats(stats, len(embedded), output_dir)

    # Return embedded chunks so the caller (FastAPI / vector_store.py) can upsert them
    return {
        "parsed_docs": parsed_docs,
        "chunks": chunks,
        "embedded_chunks": embedded,
        "stats": summary,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LexSearch Phase 1 — ingest legal PDFs into chunks + embeddings"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Directory containing source PDFs (default: data/raw)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory for chunks and stats (default: data/processed)"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["openai", "bge-m3"],
        default="openai",
        help="Embedding model to use (default: openai)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subdirectories for PDFs"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only parse and chunk; skip embedding (for testing)"
    )

    args = parser.parse_args()

    run_ingestion_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        embed_model=args.model,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
        recursive=args.recursive,
        skip_embedding=args.skip_embedding,
    )
