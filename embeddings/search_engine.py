import logging
from typing import List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSearchEngine:
    """
    Semantic search engine using sentence embeddings and FAISS.
    """

    def __init__(self, encoder=None, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = encoder if encoder is not None else SentenceTransformer(model_name)
        self.index = None
        self.documents: List[str] = []
        self._doc_to_idx: dict = {}

    def _encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if hasattr(self.model, "encode"):
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)

        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.model.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                ).to(self.model.device)
                outputs = self.model(**inputs)
                emb = outputs[0] if isinstance(outputs, tuple) else outputs
                all_embeddings.append(F.normalize(emb, dim=-1).cpu().numpy())
        return np.vstack(all_embeddings)

    def build_index(self, documents: List[str], batch_size: int = 32) -> None:
        self.documents = documents
        self._doc_to_idx = {doc: i for i, doc in enumerate(documents)}
        logger.info("Encoding %d documents...", len(documents))

        embeddings = self._encode(documents, batch_size=batch_size)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        logger.info("FAISS index built with %d vectors of dim %d.", self.index.ntotal, dimension)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self._encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        return [(self.documents[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]

    def doc_index(self, doc_text: str) -> int:
        return self._doc_to_idx[doc_text]

    def save_index(self, index_path: str = "index.faiss", docs_path: str = "documents.npy") -> None:
        if self.index is None:
            raise ValueError("No index to save. Build index first.")

        faiss.write_index(self.index, index_path)
        np.save(docs_path, np.array(self.documents))
        logger.info("Saved FAISS index to '%s' and documents to '%s'.", index_path, docs_path)

    def load_index(self, index_path: str = "index.faiss", docs_path: str = "documents.npy") -> None:
        self.index = faiss.read_index(index_path)
        self.documents = np.load(docs_path, allow_pickle=True).tolist()
        self._doc_to_idx = {doc: i for i, doc in enumerate(self.documents)}
        logger.info("Loaded FAISS index (%d vectors) and %d documents.", self.index.ntotal, len(self.documents))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    documents = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "Natural language processing involves understanding human language.",
        "Computer vision focuses on interpreting visual data.",
        "Reinforcement learning is about training agents to make decisions."
    ]

    engine = EmbeddingSearchEngine()
    engine.build_index(documents)

    for doc, score in engine.search("What is machine learning?", top_k=3):
        print(f"Score: {score:.4f} | {doc}")