"""
parser.py — PDF parsing for LexSearch
Extracts text, metadata, and structure from legal PDFs using PyMuPDF.
"""

import fitz  # PyMuPDF
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ParsedDocument:
    """Represents a fully parsed legal document."""
    doc_id: str
    filename: str
    doc_type: str                   # contract | case_law | statute | unknown
    title: str
    page_count: int
    full_text: str
    pages: list[dict]               # [{page_num, text, char_count}]
    metadata: dict
    parsed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_doc_id(filepath: str) -> str:
    """Stable SHA-256 hash of the file content → unique document ID."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _detect_doc_type(text: str, filename: str) -> str:
    """Heuristic detection of legal document type from content signals."""
    text_lower = text[:3000].lower()
    filename_lower = filename.lower()

    if any(kw in text_lower for kw in ["agreement", "contract", "whereas", "indemnif", "party a", "party b"]):
        return "contract"
    if any(kw in text_lower for kw in ["plaintiff", "defendant", "court of", "judgment", "appellant", "respondent"]):
        return "case_law"
    if any(kw in text_lower for kw in ["section", "subsection", "enacted", "statute", "legislature", "regulation"]):
        return "statute"
    if "contract" in filename_lower:
        return "contract"
    if any(kw in filename_lower for kw in ["case", "opinion", "judgment"]):
        return "case_law"
    return "unknown"


def _extract_title(doc: fitz.Document, full_text: str) -> str:
    """Try to extract a meaningful title from document metadata or first lines."""
    # 1. PDF metadata title
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title and len(meta_title) > 5:
        return meta_title

    # 2. First non-empty line of text (often the document title)
    for line in full_text.splitlines():
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            return line

    return "Untitled Document"


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, fix common PDF artifacts."""
    # Fix hyphenated line breaks (e.g. "agree-\nment" → "agreement")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize non-breaking spaces and other unicode whitespace
    text = text.replace("\xa0", " ").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # Strip lines that are pure page numbers or headers (short isolated numbers)
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\d{1,3}\s*", ln)]
    return "\n".join(lines).strip()


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_pdf(filepath: str | Path) -> ParsedDocument:
    """
    Parse a legal PDF into a structured ParsedDocument.

    Args:
        filepath: Path to the PDF file.

    Returns:
        ParsedDocument with full text, per-page data, and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    doc = fitz.open(str(filepath))

    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {filepath.name}")

    pages = []
    all_text_parts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        raw_text = page.get_text("text")          # plain text extraction
        cleaned = _clean_text(raw_text)

        pages.append({
            "page_num": page_num + 1,             # 1-indexed for humans
            "text": cleaned,
            "char_count": len(cleaned),
        })
        all_text_parts.append(cleaned)

    full_text = "\n\n".join(all_text_parts)

    if len(full_text.strip()) < 100:
        raise ValueError(
            f"Insufficient text extracted from {filepath.name}. "
            "The PDF may be scanned — consider running OCR first (see ocr_utils.py)."
        )

    doc_type = _detect_doc_type(full_text, filepath.name)
    title = _extract_title(doc, full_text)
    doc_id = _generate_doc_id(str(filepath))

    # Collect PDF metadata
    metadata = {
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "file_size_kb": round(filepath.stat().st_size / 1024, 1),
        "source_path": str(filepath),
    }

    doc.close()

    return ParsedDocument(
        doc_id=doc_id,
        filename=filepath.name,
        doc_type=doc_type,
        title=title,
        page_count=len(pages),
        full_text=full_text,
        pages=pages,
        metadata=metadata,
    )


def parse_directory(dir_path: str | Path, recursive: bool = False) -> list[ParsedDocument]:
    """
    Parse all PDFs in a directory.

    Args:
        dir_path: Directory containing PDF files.
        recursive: If True, also search subdirectories.

    Returns:
        List of successfully parsed ParsedDocuments.
    """
    dir_path = Path(dir_path)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(dir_path.glob(pattern))

    if not pdf_files:
        print(f"[parser] No PDF files found in {dir_path}")
        return []

    results = []
    errors = []

    for pdf_path in pdf_files:
        try:
            doc = parse_pdf(pdf_path)
            results.append(doc)
            print(f"[parser] ✓ {pdf_path.name} — {doc.page_count} pages, type={doc.doc_type}")
        except Exception as e:
            errors.append((pdf_path.name, str(e)))
            print(f"[parser] ✗ {pdf_path.name} — {e}")

    print(f"\n[parser] Done: {len(results)} parsed, {len(errors)} failed")
    return results
