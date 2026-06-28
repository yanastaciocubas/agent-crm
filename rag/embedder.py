import json
import os
from openai import OpenAI
from rag.vector_store import VectorStore


EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()  # uses OPENAI_API_KEY env var, same as the rest of project

FAQ_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base/faq.json")
RESOLVED_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base/resolved_tickets.json")


def load_faq_documents() -> list[dict]:
    """
    Convert FAQ entries into indexable text chunks.
    We concatenate Q+A so the chunk is semantically rich.
    """
    with open(FAQ_PATH) as f:
        faqs = json.load(f)

    documents = []
    for item in faqs:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        documents.append({
            "id": item["id"],
            "text": text,
            "metadata": {
                "source": "faq",
                "category": item["category"],
                "question": item["question"],
            },
        })

    return documents


def load_resolved_ticket_documents() -> list[dict]:
    """
    Convert resolved tickets into indexable text chunks.
    We combine the original issue + resolution so retrieval
    can match both the problem description and the fix.
    """
    with open(RESOLVED_PATH) as f:
        tickets = json.load(f)

    documents = []
    for item in tickets:
        text = (
            f"Issue: {item['original_ticket']}\n"
            f"Resolution: {item['resolution']}\n"
            f"Outcome: {item['outcome']}"
        )
        documents.append({
            "id": item["id"],
            "text": text,
            "metadata": {
                "source": "resolved_ticket",
                "category": item["category"],
                "outcome": item["outcome"],
                "customer_rating": item["customer_rating"],
            },
        })

    return documents


def _embed(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API and return list of vectors."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index(force_rebuild: bool = False):
    """
    Main entry point. Loads all knowledge base docs, embeds them,
    and indexes into Chroma. Skips if already indexed (unless force_rebuild=True).
    """
    store = VectorStore()

    if store.count() > 0 and not force_rebuild:
        print(f"[Embedder] Index already contains {store.count()} docs. Skipping rebuild.")
        print("[Embedder] Pass force_rebuild=True to re-index.")
        return store

    if force_rebuild:
        store.reset()

    print("[Embedder] Loading knowledge base documents...")
    faq_docs = load_faq_documents()
    resolved_docs = load_resolved_ticket_documents()
    all_docs = faq_docs + resolved_docs
    print(f"[Embedder] Loaded {len(faq_docs)} FAQ docs + {len(resolved_docs)} resolved ticket docs.")

    print(f"[Embedder] Embedding with OpenAI {EMBEDDING_MODEL}...")
    texts = [doc["text"] for doc in all_docs]
    embeddings = _embed(texts)

    store.add_documents(all_docs, embeddings)
    print(f"[Embedder] Done. {store.count()} documents indexed.")

    return store


def get_query_embedding(query: str) -> list[float]:
    """Embed a single query string. Used at retrieval time."""
    return _embed([query])[0]


if __name__ == "__main__":
    # Run this once to build the index:
    # python -m rag.embedder
    build_index(force_rebuild=True)