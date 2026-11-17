"""
qdrant_service.py
=================
Qdrant Cloud Vector Database Connector for Python ML microservice.
Supports cluster health checks, point upserting, and vector cosine similarity search.
"""

import os
import json
import uuid
import hashlib
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ml-service.qdrant")

class QdrantService:
    def __init__(self):
        self.url = os.environ.get("QDRANT_URL", "").rstrip("/")
        self.api_key = os.environ.get("QDRANT_API_KEY", "")
        self.collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "ticket_embeddings")
        self.vector_dim = 384
        self.timeout = 5.0
        self.collection_verified = False

    def is_configured(self) -> bool:
        return bool(self.url and len(self.url) > 0)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def to_uuid(self, text_id: str) -> str:
        """Deterministically generates a UUID string from an arbitrary text ID."""
        if not text_id:
            return "00000000-0000-0000-0000-000000000000"
        m = hashlib.md5(text_id.encode("utf-8")).hexdigest()
        return f"{m[:8]}-{m[8:12]}-{m[12:16]}-{m[16:20]}-{m[20:32]}"

    def health_check(self) -> bool:
        """Verifies connection to the Qdrant cluster."""
        if not self.is_configured():
            return False
        try:
            req = urllib.request.Request(
                f"{self.url}/collections",
                headers=self._headers(),
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def ensure_collection(self) -> bool:
        """Ensures that the collection exists with cosine similarity."""
        if self.collection_verified:
            return True
        if not self.is_configured():
            return False

        try:
            req = urllib.request.Request(
                f"{self.url}/collections/{self.collection_name}",
                headers=self._headers(),
                method="GET"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        self.collection_verified = True
                        return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    create_payload = {
                        "vectors": {
                            "size": self.vector_dim,
                            "distance": "Cosine"
                        }
                    }
                    create_req = urllib.request.Request(
                        f"{self.url}/collections/{self.collection_name}",
                        data=json.dumps(create_payload).encode("utf-8"),
                        headers=self._headers(),
                        method="PUT"
                    )
                    with urllib.request.urlopen(create_req, timeout=self.timeout) as create_resp:
                        if create_resp.status in (200, 201):
                            logger.info(f"Created Qdrant collection '{self.collection_name}'")
                            self.collection_verified = True
                            return True
                return False
        except Exception as err:
            logger.warning(f"Error ensuring Qdrant collection: {err}")
            return False

    def upsert_vectors(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]) -> bool:
        """Upserts a batch of vector points to Qdrant."""
        if not self.is_configured():
            return False

        self.ensure_collection()
        points = []
        for i in range(len(ids)):
            points.append({
                "id": self.to_uuid(ids[i]),
                "vector": embeddings[i],
                "payload": {
                    "ticket_id": ids[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "document": documents[i] if i < len(documents) else ""
                }
            })

        try:
            req = urllib.request.Request(
                f"{self.url}/collections/{self.collection_name}/points",
                data=json.dumps({"points": points}).encode("utf-8"),
                headers=self._headers(),
                method="PUT"
            )
            with urllib.request.urlopen(req, timeout=self.timeout * 2) as resp:
                return resp.status in (200, 201)
        except Exception as e:
            logger.warning(f"Failed to upsert points to Qdrant: {e}")
            return False

    def query_similar(self, query_vector: List[float], top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries nearest neighbors from Qdrant by cosine similarity."""
        if not self.is_configured():
            return []

        self.ensure_collection()
        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }
        if category:
            payload["filter"] = {
                "must": [
                    {"key": "payload.metadata.category", "match": {"value": category}}
                ]
            }

        try:
            req = urllib.request.Request(
                f"{self.url}/collections/{self.collection_name}/points/search",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                hits = data.get("result", [])

                results = []
                for hit in hits:
                    p = hit.get("payload", {})
                    results.append({
                        "ticket_id": p.get("ticket_id", hit.get("id")),
                        "similarity_score": round(float(hit.get("score", 0.0)), 4),
                        "metadata": p.get("metadata", {}),
                        "document": p.get("document", "")
                    })
                return results
        except Exception as e:
            logger.warning(f"Qdrant query error: {e}")
            return []

qdrant_service = QdrantService()
