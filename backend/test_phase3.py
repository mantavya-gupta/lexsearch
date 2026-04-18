import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import get_client, create_collection, upsert_chunks
from app.retrieval.hybrid_search import build_bm25_from_embedded
from app.generation.rag_chain import run_rag_query
from app.generation.memory import ConversationMemory

print("=" * 55)
print("LexSearch Phase 3 - Full RAG Chain Test")
print("=" * 55)

# Build the full pipeline
print("\n[1/4] Parsing and chunking...")
doc = parse_pdf("data/raw/sample1.pdf")
chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)
print(f"      {len(chunks)} chunks ready")

print("\n[2/4] Embedding chunks (this calls OpenAI)...")
embedded = embed_chunks(chunks, model="openai", cache_path=Path("data/processed/.cache/embeddings.json"))
print(f"      {len(embedded)} chunks embedded")

print("\n[3/4] Loading into Qdrant + BM25...")
client = get_client()
create_collection(client, recreate=True)
upsert_chunks(client, embedded)
bm25 = build_bm25_from_embedded(embedded, save=False)
print("      Vector store and BM25 ready")

print("\n[4/4] Running RAG queries...")
memory = ConversationMemory()

questions = [
    "What is this document about?",
    "What information is required to fill this form?",
]

for q in questions:
    print(f"\n{'─'*55}")
    result = run_rag_query(q, client, bm25)
    memory.add(q, result["answer"])
    print(f"\nQ: {q}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources used: {result['chunks_used']}")
    print(f"Top source: {result['sources'][0]['filename']} p.{result['sources'][0]['page']}")

print(f"\n{'='*55}")
print("Phase 3 complete! Full RAG pipeline working.")
print("\nNext: say 'write Phase 4 code' for the FastAPI backend!")
print("="*55)
