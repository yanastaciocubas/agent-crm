import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Thin wrapper around ChromaDB.
    Persists embeddings to disk so you only need to index once.
    """

    def __init__(self, persist_dir: str = "./rag/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="agentcrm_knowledge",
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

    def add_documents(self, documents: list[dict], embeddings: list[list[float]]):
        """
        Add a batch of documents with their precomputed embeddings.

        documents: list of dicts with keys: id, text, metadata
        embeddings: list of embedding vectors (same order as documents)
        """
        self.collection.add(
            ids=[doc["id"] for doc in documents],
            embeddings=embeddings,
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
        )
        print(f"[VectorStore] Indexed {len(documents)} documents.")

    def query(self, query_embedding: list[float], n_results: int = 3) -> list[dict]:
        """
        Retrieve the top-n most similar documents to the query embedding.
        Returns list of dicts with keys: id, text, metadata, score.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                # Chroma returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity score 0-1 for readability
                "score": round(1 - results["distances"][0][i] / 2, 4),
            })

        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Drop and recreate the collection. Use when re-indexing from scratch."""
        self.client.delete_collection("agentcrm_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="agentcrm_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")