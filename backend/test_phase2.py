import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.retrieval.vector_store import get_client, create_collection, upsert_chunks, get_collection_info
from app.retrieval.hybrid_search import BM25Index, build_bm25_from_embedded

print("=" * 55)
print("LexSearch Phase 2 - Vector Store + Hybrid Search Test")
print("=" * 55)

# Step 1: Parse real PDF
print("\n[1/4] Parsing PDF...")
doc = parse_pdf("data/raw/sample1.pdf")
print(f"      OK: {doc.filename}, {doc.page_count} pages")

# Step 2: Chunk it
print("\n[2/4] Chunking...")
chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)
print(f"      OK: {len(chunks)} chunks created")

# Step 3: Store in Qdrant (in-memory, no embeddings needed for this test)
print("\n[3/4] Setting up Qdrant (in-memory)...")
client = get_client()
create_collection(client, recreate=True)

# Simulate embedded chunks with fake vectors for testing
import random
class FakeEmbedded:
    def __init__(self, chunk):
        self.chunk = chunk
        self.embedding = [random.uniform(-1, 1) for _ in range(1536)]
        self.model = "fake"
        self.dimensions = 1536

fake_embedded = [FakeEmbedded(c) for c in chunks]
upsert_chunks(client, fake_embedded)
info = get_collection_info(client)
print(f"      OK: {info['total_chunks']} chunks in Qdrant")

# Step 4: BM25 keyword search
print("\n[4/4] Testing BM25 keyword search...")
bm25 = build_bm25_from_embedded(fake_embedded, save=False)
results = bm25.search("taxpayer identification number", top_k=3)
print(f"      OK: {len(results)} BM25 results found")
if results:
    print(f"      Top result score: {results[0]['bm25_score']}")
    print(f"      Top result preview: {results[0]['text'][:80]}...")

print("\n" + "=" * 55)
print("Phase 2 tests passed!")
print("\nNext: Set OPENAI_API_KEY to run with real embeddings")
print("Then say 'write Phase 3 code' for the RAG chain!")
print("=" * 55)
