"""
faiss_index.py
==============
FAISS Vector Search Index Manager (IndexFlatIP with L2 Normalized Vectors).

Features:
- Exact cosine similarity via Inner Product on normalized embeddings
- Bidirectional vector position <-> ticket_id mapping
- Serialization to disk via faiss.write_index / faiss.read_index
- Graceful in-memory dot-product fallback if native C++ FAISS library is missing
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger("ml-service.faiss")

class FaissIndexManager:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.id_map = []  # position -> ticket_id
        self.matrix_fallback = None  # in-memory float32 matrix for non-FAISS fallback
        self._has_native_faiss = False
        try:
            import faiss
            self._has_native_faiss = True
        except ImportError:
            self._has_native_faiss = False
            logger.warning("Native FAISS library not found. Using high-performance NumPy dot-product fallback.")

    def build_index(self, embeddings: np.ndarray, ticket_ids: list[str]):
        """
        Builds a new IndexFlatIP from normalized embeddings.
        """
        assert len(embeddings) == len(ticket_ids), "Embeddings and ticket_ids count mismatch"
        if len(embeddings) > 0:
            assert embeddings.shape[1] == self.dim, f"Embedding dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}"

        self.id_map = list(ticket_ids)

        if self._has_native_faiss:
            import faiss
            self.index = faiss.IndexFlatIP(self.dim)
            if len(embeddings) > 0:
                self.index.add(embeddings.astype(np.float32))
            logger.info(f"Built native FAISS IndexFlatIP with {self.index.ntotal} vectors.")
        else:
            self.matrix_fallback = embeddings.astype(np.float32)
            logger.info(f"Built NumPy vector index with {len(self.matrix_fallback)} vectors.")

    def save(self, index_path: str, id_map_path: str):
        """Persists the FAISS binary index and ticket ID mapping to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if self._has_native_faiss and self.index is not None:
            import faiss
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
        elif self.matrix_fallback is not None:
            np.save(index_path.replace(".bin", ".npy"), self.matrix_fallback)
            logger.info(f"Saved NumPy vector matrix to {index_path.replace('.bin', '.npy')}")

        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2)
        logger.info(f"Saved ticket ID map to {id_map_path}")

    def load(self, index_path: str, id_map_path: str) -> bool:
        """Loads FAISS index and ticket ID mapping from disk."""
        if not os.path.exists(id_map_path):
            logger.warning(f"ID map path does not exist: {id_map_path}")
            return False

        with open(id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        if self._has_native_faiss and os.path.exists(index_path):
            import faiss
            self.index = faiss.read_index(index_path)
            assert self.index.d == self.dim, f"Loaded index dimension mismatch: {self.index.d} vs {self.dim}"
            logger.info(f"Loaded native FAISS index from {index_path} ({self.index.ntotal} vectors).")
            return True
        else:
            npy_path = index_path.replace(".bin", ".npy")
            if os.path.exists(npy_path):
                self.matrix_fallback = np.load(npy_path).astype(np.float32)
                logger.info(f"Loaded vector matrix fallback from {npy_path} ({len(self.matrix_fallback)} vectors).")
                return True

        logger.warning(f"No valid FAISS index found at {index_path}")
        return False

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple[str, float, int]]:
        """
        Searches nearest neighbors using Inner Product (cosine similarity).
        Returns list of (ticket_id, similarity_score, rank).
        """
        if len(self.id_map) == 0:
            return []

        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self.id_map))

        if self._has_native_faiss and self.index is not None:
            scores, indices = self.index.search(q_vec, k)
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                if idx >= 0 and idx < len(self.id_map):
                    results.append((self.id_map[idx], float(score), rank))
            return results
        elif self.matrix_fallback is not None:
            # Exact cosine similarity via inner product: Q . M^T
            scores = np.dot(self.matrix_fallback, q_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for rank, idx in enumerate(top_indices, start=1):
                results.append((self.id_map[idx], float(scores[idx]), rank))
            return results

        return []
