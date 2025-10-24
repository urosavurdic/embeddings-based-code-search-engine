"""
Embedding-based Search Engine

Implements a simple semantic search engine using Sentence Transformers and FAISS.
This script:
- Builds a vector index for a given document collection
- Provides a search API for queries
- Supports saving/loading of the index and document list
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple


class EmbeddingSearchEngine:
    """
    A simple semantic search engine using sentence embeddings and FAISS for similarity search.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine.

        Args:
            model_name: Hugging Face model name for SentenceTransformer.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Builds the FAISS index from a list of documents.

        Args:
            documents: List of text documents.
        """
        self.documents = documents
        print(f"[INFO] Encoding {len(documents)} documents...")
        embeddings = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors of dim {dimension}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Performs semantic search over the built index.

        Args:
            query: Query string.
            top_k: Number of top results to return.

        Returns:
            List of tuples: (document_text, similarity_score)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.documents[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]
        return results

    def save_index(self, index_path: str = "index.faiss", docs_path: str = "documents.npy") -> None:
        """
        Saves the FAISS index and documents to disk.

        Args:
            index_path: Path to save FAISS index.
            docs_path: Path to save document list.
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")

        faiss.write_index(self.index, index_path)
        np.save(docs_path, np.array(self.documents))
        print(f"[INFO] Saved FAISS index to '{index_path}' and documents to '{docs_path}'.")

    def load_index(self, index_path: str = "index.faiss", docs_path: str = "documents.npy") -> None:
        """
        Loads the FAISS index and document list from disk.

        Args:
            index_path: Path to saved FAISS index.
            docs_path: Path to saved document list.
        """
        self.index = faiss.read_index(index_path)
        self.documents = np.load(docs_path, allow_pickle=True).tolist()
        print(f"[INFO] Loaded FAISS index ({self.index.ntotal} vectors) and {len(self.documents)} documents.")


if __name__ == "__main__":
    # --- Example usage ---
    documents = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "Natural language processing involves understanding human language.",
        "Computer vision focuses on interpreting visual data.",
        "Reinforcement learning is about training agents to make decisions."
    ]

    search_engine = EmbeddingSearchEngine()
    search_engine.build_index(documents)

    query = "What is machine learning?"
    results = search_engine.search(query, top_k=3)

    print("\nTop results:")
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: {doc}")

    # Optional: save & reload demonstration
    search_engine.save_index()
    new_engine = EmbeddingSearchEngine()
    new_engine.load_index()

    print("\nAfter reloading index:")
    results = new_engine.search("How do computers see images?", top_k=3)
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: {doc}")