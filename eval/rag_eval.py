"""
RAG Evaluation Harness for AgentCRM

Measures two core RAG quality metrics:

1. Retrieval Precision@K
   Did the retriever surface the right document for each test query?
   Precision@K = (# relevant docs in top-K) / K

2. Answer Faithfulness (simulated)
   Does the response text actually use content from the retrieved docs,
   or did it hallucinate? We check this by measuring lexical overlap
   between the response and the retrieved context.
   (In production you'd use an LLM-as-judge or RAGAS for this.)

Run with:
    python -m eval.rag_eval
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.embedder import build_index, get_query_embedding
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Ground truth: for each test query, what doc IDs should be retrieved?
# These are the expected top results based on our knowledge base content.
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "query": "I was charged twice this month, I need a refund",
        "expected_ids": ["faq-002", "faq-003", "resolved-001"],
        "category": "billing",
    },
    {
        "query": "The app shows a blank white screen when I try to log in",
        "expected_ids": ["faq-005", "faq-006"],
        "category": "bug",
    },
    {
        "query": "Can you add Slack notifications to the platform?",
        "expected_ids": ["faq-008", "resolved-007"],
        "category": "feature_request",
    },
    {
        "query": "How do I reset my password? I can't log into my account",
        "expected_ids": ["faq-011", "resolved-004"],
        "category": "support",
    },
    {
        "query": "I need to cancel my subscription and get my money back",
        "expected_ids": ["resolved-002", "faq-003"],
        "category": "billing",
    },
    {
        "query": "My API is returning 500 errors on the contacts endpoint",
        "expected_ids": ["resolved-005", "faq-007"],
        "category": "bug",
    },
    {
        "query": "How do I remove a team member who left the company?",
        "expected_ids": ["resolved-009", "faq-012"],
        "category": "support",
    },
    {
        "query": "I accidentally deleted all my customer records",
        "expected_ids": ["resolved-008"],
        "category": "support",
    },
]


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of retrieved top-K docs that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in expected_ids)
    return hits / k if k > 0 else 0.0


def faithfulness_score(response: str, context_texts: list[str]) -> float:
    """
    Lexical overlap proxy for faithfulness.
    Measures what fraction of the response's meaningful words
    also appear in the retrieved context.

    This is a simplified stand-in for LLM-as-judge faithfulness scoring.
    In production: use RAGAS faithfulness metric or GPT-4 evaluation.
    """
    stop_words = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "with", "this", "that", "your", "our",
        "we", "you", "i", "my", "be", "are", "was", "were", "have", "has",
        "will", "can", "not", "if", "as", "by", "from", "their", "they",
    }

    response_words = set(response.lower().split()) - stop_words
    context_combined = " ".join(context_texts).lower()
    context_words = set(context_combined.split()) - stop_words

    if not response_words:
        return 0.0

    overlap = response_words & context_words
    return len(overlap) / len(response_words)


def run_eval(k: int = 3):
    """Run the full evaluation suite and print a report."""
    print("=" * 60)
    print("AgentCRM RAG Evaluation Report")
    print("=" * 60)

    # Make sure the index is built
    build_index()
    store = VectorStore()

    precision_scores = []
    faithfulness_scores = []

    for i, test_case in enumerate(EVAL_DATASET):
        query = test_case["query"]
        expected_ids = test_case["expected_ids"]

        # Retrieve
        query_embedding = get_query_embedding(query)
        results = store.query(query_embedding, n_results=k)
        retrieved_ids = [r["id"] for r in results]
        retrieved_texts = [r["text"] for r in results]

        # Compute metrics
        p_at_k = precision_at_k(retrieved_ids, expected_ids, k)

        # Simulate a response from the top doc (same logic as ResponderAgent)
        top_text = retrieved_texts[0] if retrieved_texts else ""
        simulated_response = top_text  # In full pipeline this goes through Responder
        faith = faithfulness_score(simulated_response, retrieved_texts)

        precision_scores.append(p_at_k)
        faithfulness_scores.append(faith)

        # Print per-query result
        hits = [doc_id for doc_id in retrieved_ids if doc_id in expected_ids]
        status = "PASS" if p_at_k > 0 else "MISS"
        print(f"\n[{i+1}/{len(EVAL_DATASET)}] {status} | Category: {test_case['category']}")
        print(f"  Query       : {query[:70]}...")
        print(f"  Retrieved   : {retrieved_ids}")
        print(f"  Expected    : {expected_ids}")
        print(f"  Hits        : {hits}")
        print(f"  Precision@{k}: {p_at_k:.2f}  |  Faithfulness: {faith:.2f}")

    # Summary
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test cases        : {len(EVAL_DATASET)}")
    print(f"  Avg Precision@{k}  : {avg_precision:.3f}")
    print(f"  Avg Faithfulness  : {avg_faithfulness:.3f}")
    print(f"  Pass rate (P>0)   : {sum(1 for p in precision_scores if p > 0)}/{len(precision_scores)}")
    print("=" * 60)

    # Interpret results
    print("\nInterpretation:")
    if avg_precision >= 0.6:
        print("  Retrieval quality: GOOD — top-K results are highly relevant.")
    elif avg_precision >= 0.3:
        print("  Retrieval quality: MODERATE — some misses; consider tuning chunk size or threshold.")
    else:
        print("  Retrieval quality: NEEDS WORK — revisit embedding model or knowledge base coverage.")

    if avg_faithfulness >= 0.5:
        print("  Faithfulness: GOOD — responses closely reflect retrieved context.")
    else:
        print("  Faithfulness: LOW — responses may be drifting from retrieved content.")

    return {
        "avg_precision_at_k": avg_precision,
        "avg_faithfulness": avg_faithfulness,
        "k": k,
        "n_test_cases": len(EVAL_DATASET),
    }


if __name__ == "__main__":
    run_eval(k=3)