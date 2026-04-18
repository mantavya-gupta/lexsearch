"""
chunker.py — Smart legal document chunking for LexSearch
Splits parsed legal documents into retrieval-ready chunks, preserving
clause boundaries and injecting rich metadata into each chunk.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from .parser import ParsedDocument


@dataclass
class Chunk:
    """A single retrievable chunk with full provenance metadata."""
    chunk_id: str
    doc_id: str
    doc_type: str
    filename: str
    title: str
    text: str                   # The actual text to embed and retrieve
    char_count: int
    token_estimate: int         # Rough estimate: chars / 4
    chunk_index: int            # Position of this chunk in document
    page_start: int             # First page this chunk spans
    page_end: int               # Last page this chunk spans
    clause_type: Optional[str]  # indemnity | liability | termination | payment | etc.
    section_header: Optional[str]
    metadata: dict = field(default_factory=dict)


# ── Legal clause classifier ───────────────────────────────────────────────────

# Maps regex patterns to clause type labels
CLAUSE_PATTERNS: list[tuple[str, str]] = [
    (r"\bindemnif", "indemnity"),
    (r"\bliabilit", "liability"),
    (r"\btermina(te|tion)", "termination"),
    (r"\bconfidential", "confidentiality"),
    (r"\bpayment|compensation|fee[s]?\b", "payment"),
    (r"\bintellectual property|ip rights|copyright|patent", "intellectual_property"),
    (r"\bgoverning law|jurisdiction|dispute|arbitration", "dispute_resolution"),
    (r"\bwarrant(y|ies)|represent", "warranties"),
    (r"\bforce majeure", "force_majeure"),
    (r"\bprivacy|data protection|gdpr|personal data", "data_privacy"),
    (r"\bdefin(e|ition)", "definitions"),
    (r"\bappeal|remand|affirm|reverse", "court_ruling"),
    (r"\bholding:|held:|court held", "court_holding"),
    (r"\bpursuant to section|§\s*\d", "statutory_reference"),
]

# Section header patterns (lines that start a new clause/section)
SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:ARTICLE|SECTION|CLAUSE)\s+[IVXLCDM\d]+[.):]"   # ARTICLE IV: / SECTION 2.
    r"|(?:\d+\.)+\s+[A-Z]"                                # 3.1 Payment Terms
    r"|[A-Z][A-Z\s]{4,40}(?:\s*[:.])\s*$"                # ALL CAPS HEADING:
    r")",
    re.MULTILINE,
)


def _detect_clause_type(text: str) -> Optional[str]:
    """Return the first matching clause type label, or None."""
    text_lower = text.lower()
    for pattern, label in CLAUSE_PATTERNS:
        if re.search(pattern, text_lower):
            return label
    return None


def _extract_section_header(text: str) -> Optional[str]:
    """Return the first section header found in the chunk text, if any."""
    match = SECTION_HEADER_RE.search(text)
    if match:
        header = match.group(0).strip()
        return header[:80] if len(header) > 80 else header
    return None


# ── Core chunking logic ────────────────────────────────────────────────────────

def _split_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Recursive character-level splitting that respects natural boundaries.
    Priority order: double-newline → single-newline → period → space.
    This preserves paragraph and sentence integrity better than hard splits.
    """
    # If text fits in a single chunk, return as-is
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Find the best split point within the window
            best_sep_pos = -1
            for sep in separators:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    best_sep_pos = pos + len(sep)
                    break

            if best_sep_pos > start:
                end = best_sep_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, accounting for overlap
        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def _map_chunk_to_pages(
    chunk_text: str,
    pages: list[dict],
) -> tuple[int, int]:
    """
    Find which pages a chunk spans by looking for its content in page texts.
    Returns (page_start, page_end) as 1-indexed page numbers.
    """
    # Use first 80 chars of chunk as a search key
    search_key = chunk_text[:80].strip()

    first_page, last_page = 1, len(pages)

    for page in pages:
        if search_key in page["text"]:
            first_page = page["page_num"]
            break

    # For last page: use last 80 chars
    search_key_end = chunk_text[-80:].strip()
    for page in reversed(pages):
        if search_key_end in page["text"]:
            last_page = page["page_num"]
            break

    return first_page, last_page


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = 512 * 4,      # ~512 tokens (1 token ≈ 4 chars)
    chunk_overlap: int = 50 * 4,    # ~50 token overlap
) -> list[Chunk]:
    """
    Chunk a ParsedDocument into retrieval-ready Chunk objects.

    Args:
        parsed_doc: Output from parser.parse_pdf()
        chunk_size:    Target chunk size in characters (~512 tokens default)
        chunk_overlap: Overlap between adjacent chunks in characters

    Returns:
        List of Chunk objects with full metadata, sorted by chunk_index.
    """
    raw_chunks = _split_into_chunks(
        parsed_doc.full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < 50:
            continue  # Skip near-empty chunks (page headers, stray numbers, etc.)

        page_start, page_end = _map_chunk_to_pages(chunk_text, parsed_doc.pages)

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=parsed_doc.doc_id,
            doc_type=parsed_doc.doc_type,
            filename=parsed_doc.filename,
            title=parsed_doc.title,
            text=chunk_text,
            char_count=len(chunk_text),
            token_estimate=len(chunk_text) // 4,
            chunk_index=idx,
            page_start=page_start,
            page_end=page_end,
            clause_type=_detect_clause_type(chunk_text),
            section_header=_extract_section_header(chunk_text),
            metadata={
                **parsed_doc.metadata,
                "parsed_at": parsed_doc.parsed_at,
            },
        )
        chunks.append(chunk)

    print(
        f"[chunker] {parsed_doc.filename} → {len(chunks)} chunks "
        f"(avg {sum(c.token_estimate for c in chunks) // max(len(chunks), 1)} tokens each)"
    )
    return chunks


def chunk_documents(
    parsed_docs: list[ParsedDocument],
    chunk_size: int = 512 * 4,
    chunk_overlap: int = 50 * 4,
) -> list[Chunk]:
    """Chunk a list of ParsedDocuments. Returns all chunks flat."""
    all_chunks = []
    for doc in parsed_docs:
        try:
            all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
        except Exception as e:
            print(f"[chunker] ✗ Failed to chunk {doc.filename}: {e}")
    print(f"\n[chunker] Total chunks produced: {len(all_chunks)}")
    return all_chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Return summary statistics about a list of chunks — useful for README/eval."""
    if not chunks:
        return {}
    token_counts = [c.token_estimate for c in chunks]
    clause_dist = {}
    for c in chunks:
        key = c.clause_type or "unclassified"
        clause_dist[key] = clause_dist.get(key, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_tokens": round(sum(token_counts) / len(token_counts)),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "unique_docs": len({c.doc_id for c in chunks}),
        "clause_distribution": clause_dist,
    }
