from rag.embedder import get_query_embedding
from rag.vector_store import VectorStore


# Minimum similarity score to include a result.
# Below this threshold the result is probably noise.
RELEVANCE_THRESHOLD = 0.30

# How many docs to retrieve per ticket.
TOP_K = 3


class RetrieverAgent:
    """
    Retrieves the most relevant knowledge base documents for a given ticket.
    Runs before the ResponderAgent so the response can be grounded in
    real company knowledge rather than generic LLM output.

    Retrieved context is stored in ticket["rag_context"] as a list of dicts,
    each with: id, text, metadata, score.
    """

    def __init__(self):
        self.store = VectorStore()

        if self.store.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Run `python -m rag.embedder` first to build the index."
            )

    def retrieve(self, ticket: dict, min_score: float = RELEVANCE_THRESHOLD) -> dict:

        """
        Embed the ticket text, query Chroma, attach top-k results to ticket.
        """
        query = ticket.get("message", "")
        if not query:
            ticket["rag_context"] = []
            return ticket

        query_embedding = get_query_embedding(query)
        results = self.store.query(query_embedding, n_results=TOP_K)

        # Filter out low-relevance results
        filtered = [r for r in results if r["score"] >= min_score]

        ticket["rag_context"] = filtered
        ticket["rag_sources"] = [r["id"] for r in filtered]

        if filtered:
            top = filtered[0]
            print(
                f"[RetrieverAgent] Top match for ticket {ticket['id']}: "
                f"{top['id']} (score={top['score']}, source={top['metadata']['source']})"
            )
        else:
            print(f"[RetrieverAgent] No relevant context found for ticket {ticket['id']}.")

        return ticket