"""
rag_chain.py — Core RAG chain for LexSearch
Uses Groq (free) for LLM generation + sentence-transformers for embeddings (free, local)
"""

import os
from groq import Groq
from .prompts import LEGAL_RAG_PROMPT, QUERY_EXPANSION_PROMPT
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank, format_context

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
CHAT_MODEL = "llama-3.1-8b-instant"


def embed_query(text, _model=[None]):
    """
    Embed a query using sentence-transformers (free, local, runs on Mac M-series).
    Model downloads once (~90MB), then runs offline.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def expand_query(question):
    """Generate 3 alternative phrasings using Groq LLaMA3 (free)."""
    try:
        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=question)}],
            temperature=0.3,
            max_tokens=200,
        )
        alternatives = response.choices[0].message.content.strip().split("\n")
        alternatives = [q.strip() for q in alternatives if q.strip()]
        print(f"[rag_chain] Query expanded to {len(alternatives)+1} variants")
        return [question] + alternatives[:3]
    except Exception as e:
        print(f"[rag_chain] Query expansion failed: {e}. Using original.")
        return [question]


def generate_answer(question, context):
    """Call Groq LLaMA3 with the grounded legal prompt."""
    prompt = LEGAL_RAG_PROMPT.format(context=context, question=question)
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def run_rag_query(
    question,
    qdrant_client,
    bm25_index,
    top_k_retrieve=20,
    top_k_rerank=5,
    use_query_expansion=True,
):
    """
    Full RAG pipeline for a single question.
    Returns answer, sources, and metadata.
    """
    print(f"\n[rag_chain] Question: {question}")

    # Step 1: Query expansion
    queries = expand_query(question) if use_query_expansion else [question]

    # Step 2: Embed and retrieve
    all_results = {}
    for q in queries:
        embedding = embed_query(q)
        results = hybrid_search(
            query=q,
            query_embedding=embedding,
            qdrant_client=qdrant_client,
            bm25_index=bm25_index,
            top_k=top_k_retrieve,
        )
        for r in results:
            cid = r["chunk_id"]
            if cid not in all_results or r["combined_score"] > all_results[cid]["combined_score"]:
                all_results[cid] = r

    candidates = sorted(all_results.values(), key=lambda x: x["combined_score"], reverse=True)
    candidates = candidates[:top_k_retrieve]
    print(f"[rag_chain] Retrieved {len(candidates)} unique chunks")

    # Step 3: Rerank
    reranked = rerank(question, candidates, top_k=top_k_rerank)
    print(f"[rag_chain] Reranked to top {len(reranked)} chunks")

    # Step 4: Format context and generate
    context = format_context(reranked)
    print(f"[rag_chain] Generating answer with Groq LLaMA3...")
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "filename": r.get("filename"),
                "page": r.get("page_start"),
                "clause_type": r.get("clause_type"),
                "score": r.get("relevance_score", r.get("combined_score")),
                "preview": r["text"][:150] + "...",
            }
            for r in reranked
        ],
        "chunks_retrieved": len(candidates),
        "chunks_used": len(reranked),
    }
