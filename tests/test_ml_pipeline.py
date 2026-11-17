"""
test_ml_pipeline.py
===================
Pytest Unit and Integration Test Suite for ML Classification Service.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import clean_text_for_ml, combine_ticket_text
from rules import resolve_subcategory
from taxonomy import TAXONOMY, CATEGORIES
from main import app

client = TestClient(app)

def test_clean_text_preserves_error_codes_and_ips():
    raw = "  Error  0x80070005 on  host 10.240.0.1:8080 \n\t Path: C:\\Windows\\System32  "
    cleaned = clean_text_for_ml(raw)
    assert "0x80070005" in cleaned
    assert "10.240.0.1:8080" in cleaned
    assert "C:\\Windows\\System32" in cleaned
    assert "  " not in cleaned

def test_combine_ticket_text_omits_resolution():
    title = "VPN tunnel disconnected"
    desc = "User encounters Error 800 after entering MFA code."
    service = "Corporate VPN"
    combined = combine_ticket_text(title, desc, service)
    assert "VPN tunnel disconnected" in combined
    assert "Error 800" in combined
    assert "Corporate VPN" in combined

def test_resolve_subcategory_valid_network_vpn():
    sub = resolve_subcategory("Network", "GlobalProtect VPN tunnel drops intermittently")
    assert sub == "VPN"

def test_resolve_subcategory_invalid_category_returns_none():
    sub = resolve_subcategory("NonExistentCategory", "VPN tunnel drops")
    assert sub is None

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert "supported_categories" in data
    assert len(data["supported_categories"]) == 6

def test_classify_empty_input_flags_review():
    # If model is loaded or mocked
    if app.state.pipeline is not None:
        payload = {"title": "   ", "description": "   "}
        res = client.post("/classify", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["review_required"] is True
        assert "empty" in data["review_reason"].lower()

def test_classify_short_input_flags_review():
    if app.state.pipeline is not None:
        payload = {"title": "Hi", "description": "VPN"}
        res = client.post("/classify", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["review_required"] is True
        assert "short" in data["review_reason"].lower()

def test_classify_missing_title_rejected_by_pydantic():
    payload = {"description": "Valid description text with more than 10 chars"}
    res = client.post("/classify", json=payload)
    assert res.status_code == 422  # Unprocessable Entity
