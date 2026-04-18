"""
reranker.py — Cohere re-ranking for LexSearch
Takes top-20 hybrid search results, re-scores them using a cross-encoder
model for 30%+ improvement in answer faithfulness.
"""

import os


def rerank(query, candidates, top_k=5, model="rerank-english-v3.0"):
    """
    Re-rank candidate chunks using Cohere's cross-encoder.

    Args:
        query:      The user's question
        candidates: List of result dicts from hybrid_search()
        top_k:      Number of final results to return (default 5)
        model:      Cohere rerank model name

    Returns:
        Top-k re-ranked results with relevance_score added.
    """
    api_key = os.getenv("COHERE_API_KEY")

    # Fallback: if no Cohere key, return top_k from candidates as-is
    if not api_key:
        print("[reranker] No COHERE_API_KEY set. Returning top candidates without re-ranking.")
        return candidates[:top_k]

    try:
        import cohere
        co = cohere.Client(api_key)

        documents = [c["text"] for c in candidates]

        response = co.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        reranked = []
        for result in response.results:
            candidate = candidates[result.index].copy()
            candidate["relevance_score"] = round(result.relevance_score, 4)
            reranked.append(candidate)

        print(f"[reranker] Re-ranked {len(candidates)} -> {len(reranked)} results")
        return reranked

    except Exception as e:
        print(f"[reranker] Cohere re-ranking failed: {e}. Returning top candidates.")
        return candidates[:top_k]


def format_context(reranked_results):
    """
    Format re-ranked results into a clean context string for the LLM prompt.
    Includes source citation info for each chunk.
    """
    context_parts = []
    for i, result in enumerate(reranked_results):
        source = f"{result.get('filename', 'unknown')} (page {result.get('page_start', '?')})"
        clause = result.get("clause_type", "general")
        context_parts.append(
            f"[Source {i+1}: {source} | Clause: {clause}]\n{result['text']}"
        )
    return "\n\n---\n\n".join(context_parts)
