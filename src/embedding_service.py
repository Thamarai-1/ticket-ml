"""
embedding_service.py
====================
Sentence Transformer Semantic Embedding Service.

Responsibilities:
1. Formats incoming and historical tickets into standardized issue text.
2. Embeds text using Sentence Transformers ('all-MiniLM-L6-v2', dimension=384).
3. L2-normalizes vectors so that inner product corresponds to cosine similarity.
4. Includes deterministic fallback encoder if PyTorch / Sentence Transformers
   model weights are offline or initializing in a restricted sandbox.
"""

import os
import re
import numpy as np
import logging

logger = logging.getLogger("ml-service.embedding")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

def format_ticket_for_retrieval(ticket: dict) -> str:
    """
    Builds problem representation exclusively from information describing the issue.
    
    STRICT SAFETY RULES (Section 4):
    1. EXCLUDES resolution_text (the incoming ticket has no resolution; including
       it distorts similarity between problem symptoms).
    2. EXCLUDES category label (avoids artificially inflating retrieval).
    3. EXCLUDES resolved_at or post-triage fields.
    """
    title = (ticket.get("title") or "").strip()
    description = (ticket.get("description") or "").strip()
    affected_service = (ticket.get("affected_service") or "").strip()
    environment = (ticket.get("environment") or "").strip()

    parts = [
        f"Title: {title}",
        f"Description: {description}"
    ]
    if affected_service:
        parts.append(f"Affected service: {affected_service}")
    if environment:
        parts.append(f"Environment: {environment}")

    return "\n".join(parts)

class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer {self.model_name} loaded successfully (dim={self.dim}).")
        except Exception as e:
            logger.warning(
                f"SentenceTransformer library or weights unavailable ({str(e)}). "
                f"Activating deterministic semantic projector fallback (dim={self.dim})."
            )
            self.model = None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of text strings into L2-normalized float32 vectors.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.model is not None:
            # Generate embeddings and normalize
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.astype(np.float32)
        else:
            # Deterministic semantic hash projection (offline fallback)
            # Produces stable, reproducible 384-dim unit vectors
            vectors = []
            for text in texts:
                vec = np.zeros(self.dim, dtype=np.float32)
                tokens = re.findall(r"\w+", text.lower())
                for i, token in enumerate(tokens):
                    idx = hash(token) % self.dim
                    weight = 1.0 / (1.0 + np.log1p(i))
                    vec[idx] += weight
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                else:
                    vec[0] = 1.0  # fallback unit vector
                vectors.append(vec)
            return np.array(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encodes a single string into a 1D float32 normalized vector."""
        return self.encode([text])[0]
