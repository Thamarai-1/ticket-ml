"""
test_reliability.py
===================
Pytest Test Suite for Phase 6 Reliability, Grounding, and Safety Safeguards.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evidence_service import is_evidence_eligible, detect_evidence_contradictions, extract_resolution_steps
from grounding_validator import GroundingValidator
from schemas import RecommendationResponse, RecommendationItem, EvidenceItem, RecommendationConfidence
from main import app

client = TestClient(app)

def test_unresolved_ticket_disqualification():
    candidate = {
        "ticket_id": "INC-NEW-01",
        "title": "VPN drops",
        "status": "In Progress",  # Unresolved status!
        "resolution_text": "Troubleshooting with user"
    }
    is_ok, reason = is_evidence_eligible(candidate, {"ticket_id": "INC-QUERY"})
    assert is_ok is False
    assert "status is 'In Progress'" in reason

def test_placeholder_resolution_disqualification():
    candidate = {
        "ticket_id": "INC-PLACEHOLDER",
        "status": "Resolved",
        "resolution_text": "Fixed"  # Less than 10 characters / placeholder!
    }
    is_ok, reason = is_evidence_eligible(candidate, {"ticket_id": "INC-QUERY"})
    assert is_ok is False
    assert "placeholder" in reason.lower() or "incomplete" in reason.lower()

def test_access_level_restriction():
    candidate = {
        "ticket_id": "INC-RESTRICTED",
        "status": "Resolved",
        "resolution_text": "Detailed confidential security forensic steps.",
        "data_access_level": "restricted"
    }
    is_ok, reason = is_evidence_eligible(candidate, {"ticket_id": "INC-QUERY"}, user_access_level="internal")
    assert is_ok is False
    assert "exceeds user clearance" in reason

def test_contradiction_detection():
    candidate_a = {
        "ticket_id": "INC-A",
        "resolution_text": "Upgraded network client to version 6.2 to resolve timeout."
    }
    candidate_b = {
        "ticket_id": "INC-B",
        "resolution_text": "Downgraded network client to version 5.4 to resolve incompatibility."
    }
    has_conflict, reasons = detect_evidence_contradictions([candidate_a, candidate_b])
    assert has_conflict is True
    assert len(reasons) >= 1
    assert "upgrade" in reasons[0] and "downgrade" in reasons[0]

def test_grounding_validator_rejects_hallucinated_source_id():
    catalog = {
        "INC-1001": {
            "ticket_id": "INC-1001",
            "resolution_text": "Reset network adapter settings in Windows Device Manager."
        }
    }
    validator = GroundingValidator(authorized_catalog=catalog)

    # Synthetic response that hallucinated INC-9999
    rec_item = RecommendationItem(
        recommendation_id="REC-001",
        title="Reset adapter",
        steps=["Reset network adapter."],
        evidence=[
            EvidenceItem(
                source_id="INC-9999",  # Hallucinated ID!
                excerpt="Reset network adapter settings.",
                relevance_reason="Similar"
            )
        ]
    )
    resp = RecommendationResponse(
        recommendation_status="suggested",
        recommendations=[rec_item],
        confidence=RecommendationConfidence(level="high", basis="Test"),
        inference_latency_ms=10.0
    )

    is_grounded, errors, validated_resp = validator.validate(resp, eligible_ticket_ids=["INC-1001"])
    assert is_grounded is False
    assert any("Hallucinated citation" in err for err in errors)
    assert len(validated_resp.recommendations) == 0

def test_fastapi_recommend_endpoint_full():
    payload = {
        "ticket_id": "INC-2026-TEST",
        "title": "Software installation error 0x80070005 Access is denied during CAD install",
        "description": "AutoCAD 2026 deployment fails via Company Portal with exit code 0x80070005.",
        "affected_service": "Endpoint Management (Intune)",
        "environment": "Production",
        "top_k": 3
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "recommendations" in data
    assert "confidence" in data
    assert data["review_required"] is True
