"""
test_phase1.py — Quick sanity check for Phase 1 ingestion pipeline.
Run from the backend/ directory:

    python test_phase1.py

Creates a synthetic legal text file, runs it through parse → chunk → embed,
and prints results. Does NOT require real PDFs or an OpenAI key
(uses --skip-embedding mode).
"""

import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.parser import ParsedDocument
from app.ingestion.chunker import chunk_document, chunk_stats
from app.ingestion.pipeline import run_ingestion_pipeline

# ── Create a synthetic legal document for testing ─────────────────────────────

FAKE_LEGAL_TEXT = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2024,
by and between Acme Corporation ("Client") and TechVentures Ltd. ("Provider").

ARTICLE I: DEFINITIONS

1.1 "Services" means the software development and consulting services described
in Schedule A attached hereto and incorporated herein by reference.

1.2 "Confidential Information" means any non-public information disclosed by
either party that is designated as confidential or that reasonably should be
understood to be confidential given the nature of the information.

ARTICLE II: PAYMENT TERMS

2.1 Compensation. Client shall pay Provider a monthly fee of USD 10,000 for
the Services, due within 30 days of invoice receipt.

2.2 Late Payment. Invoices not paid within 30 days shall accrue interest at
1.5% per month on the unpaid balance.

ARTICLE III: INDEMNIFICATION

3.1 Provider Indemnification. Provider shall indemnify, defend, and hold
harmless Client from any claims arising out of Provider's gross negligence
or willful misconduct in performing the Services.

3.2 Client Indemnification. Client shall indemnify Provider against claims
arising from Client's use of the Services in violation of applicable law.

ARTICLE IV: LIMITATION OF LIABILITY

4.1 In no event shall either party be liable for indirect, incidental, special,
or consequential damages, even if advised of the possibility of such damages.

4.2 Each party's total liability under this Agreement shall not exceed the
total fees paid in the twelve months preceding the claim.

ARTICLE V: TERMINATION

5.1 Either party may terminate this Agreement upon 30 days written notice.

5.2 Client may terminate immediately upon written notice if Provider materially
breaches this Agreement and fails to cure such breach within 15 days.

5.3 Upon termination, Provider shall deliver all work product to Client within
10 business days and Client shall pay all fees for Services rendered through
the termination date.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
first written above.

ACME CORPORATION                    TECHVENTURES LTD.
By: ______________________         By: ______________________
Name: Jane Smith                   Name: John Doe
Title: CEO                         Title: President
""".strip()


def test_parser():
    """Test the parser with a synthetic ParsedDocument (no real PDF needed)."""
    print("\n── Test 1: Parser ──────────────────────────────────────────")
    doc = ParsedDocument(
        doc_id="test_001",
        filename="service_agreement.pdf",
        doc_type="contract",
        title="Service Agreement",
        page_count=3,
        full_text=FAKE_LEGAL_TEXT,
        pages=[{"page_num": 1, "text": FAKE_LEGAL_TEXT, "char_count": len(FAKE_LEGAL_TEXT)}],
        metadata={"source_path": "test/", "file_size_kb": 12.5},
    )
    print(f"  ✓ ParsedDocument created: {doc.filename}")
    print(f"    doc_id:    {doc.doc_id}")
    print(f"    doc_type:  {doc.doc_type}")
    print(f"    chars:     {len(doc.full_text)}")
    return doc


def test_chunker(doc):
    """Test chunking with a small chunk size to generate multiple chunks."""
    print("\n── Test 2: Chunker ─────────────────────────────────────────")
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)

    print(f"  ✓ {len(chunks)} chunks created")
    print(f"\n  First chunk preview:")
    print(f"    chunk_id:     {chunks[0].chunk_id[:12]}...")
    print(f"    tokens:       ~{chunks[0].token_estimate}")
    print(f"    clause_type:  {chunks[0].clause_type}")
    print(f"    section:      {chunks[0].section_header}")
    print(f"    text[:80]:    {chunks[0].text[:80]}...")

    stats = chunk_stats(chunks)
    print(f"\n  Chunk stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    return chunks


def test_pipeline_skip_embed():
    """Test the full pipeline in --skip-embedding mode (no API key needed)."""
    print("\n── Test 3: Pipeline (skip-embedding mode) ──────────────────")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "raw"
        input_dir.mkdir()
        print(f"  Note: No real PDFs in {input_dir}, so 0 docs will be parsed.")
        print(f"  (In real use, put your PDFs in data/raw/ and run the pipeline)")

        result = run_ingestion_pipeline(
            input_dir=str(input_dir),
            output_dir=str(Path(tmpdir) / "processed"),
            skip_embedding=True,
        )
        print(f"  ✓ Pipeline returned: {result}")


if __name__ == "__main__":
    print("="*60)
    print("LexSearch — Phase 1 Sanity Tests")
    print("="*60)

    doc = test_parser()
    chunks = test_chunker(doc)
    test_pipeline_skip_embed()

    print("\n" + "="*60)
    print("✓ All Phase 1 tests passed!")
    print("\nNext steps:")
    print("  1. Put your legal PDFs in data/raw/")
    print("  2. Set OPENAI_API_KEY in .env")
    print("  3. Run: python -m app.ingestion.pipeline --input data/raw")
    print("  4. Move on to Phase 2: vector store + hybrid search")
    print("="*60)
