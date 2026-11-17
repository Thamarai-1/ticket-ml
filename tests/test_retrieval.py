"""
test_retrieval.py
=================
Pytest Test Suite for Phase 3 Hybrid Retrieval and Duplicate Detection.
"""

import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from embedding_service import EmbeddingService, format_ticket_for_retrieval
from faiss_index import FaissIndexManager
from bm25_service import BM25Manager, tokenize_for_bm25
from hybrid_retriever import HybridRetriever, extract_technical_tokens, evaluate_duplicate_policy
from main import app

client = TestClient(app)

def test_format_ticket_for_retrieval_excludes_resolution():
    ticket = {
        "title": "VPN connection drops",
        "description": "User encounters error 800",
        "affected_service": "Corporate VPN",
        "environment": "Production",
        "resolution_text": "Updated GlobalProtect to 6.1.2",
        "category": "Network"
    }
    formatted = format_ticket_for_retrieval(ticket)
    assert "Title: VPN connection drops" in formatted
    assert "Description: User encounters error 800" in formatted
    assert "Affected service: Corporate VPN" in formatted
    assert "Updated GlobalProtect" not in formatted
    assert "Category:" not in formatted

def test_embedding_normalization():
    emb_service = EmbeddingService()
    vec = emb_service.encode_one("GlobalProtect VPN tunnel dropped unexpectedly")
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0

def test_bm25_tokenization_preserves_hex():
    tokens = tokenize_for_bm25("Installer failed with error 0x80070005 on host 10.240.0.1")
    assert "0x80070005" in tokens
    assert "10.240.0.1" in tokens

def test_faiss_build_and_search():
    mgr = FaissIndexManager(dim=384)
    embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize rows
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ticket_ids = [f"INC-100{i}" for i in range(5)]

    mgr.build_index(embeddings, ticket_ids)
    results = mgr.search(embeddings[0], top_k=3)
    assert len(results) == 3
    assert results[0][0] == "INC-1000"
    assert pytest.approx(results[0][1], rel=1e-3) == 1.0  # Perfect self cosine similarity

def test_duplicate_policy_detection():
    query = {
        "title": "VPN disconnects with error 800",
        "description": "User encounters Error 800 on gateway 10.240.0.1",
        "affected_service": "Corporate VPN",
        "environment": "Production"
    }
    candidate = {
        "title": "GlobalProtect VPN connection drops repeatedly",
        "description": "Tunnel drops with Error 800 on gateway 10.240.0.1",
        "affected_service": "Corporate VPN",
        "environment": "Production"
    }
    is_dup, conf, reason = evaluate_duplicate_policy(query, candidate, semantic_score=0.89, bm25_score=9.5)
    assert is_dup is True
    assert conf >= 0.70
    assert "Error 800" in reason or "10.240.0.1" in reason or "VPN" in reason

def test_fastapi_retrieve_endpoint():
    payload = {
        "title": "VPN disconnects after login",
        "description": "Corporate VPN connection drops immediately after authentication on gateway 10.240.0.1.",
        "affected_service": "Corporate VPN",
        "environment": "Production",
        "top_k": 3
    }
    res = client.post("/retrieve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    assert data["retrieval_method"] == "hybrid"
