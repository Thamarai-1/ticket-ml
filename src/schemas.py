"""
schemas.py
==========
Pydantic Request and Response Schemas for FastAPI ML Service (Phases 2 & 3).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# --- Phase 2: Classification Schemas ---

class TicketClassificationRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, example="INC-2001")
    title: str = Field(..., min_length=1, max_length=300, example="Unable to access corporate VPN")
    description: str = Field(..., min_length=1, max_length=10000, example="VPN authentication succeeds but drops immediately with error 800.")
    affected_service: Optional[str] = Field(None, max_length=200, example="Corporate VPN")

class ClassificationDetails(BaseModel):
    category: Optional[str] = Field(None, example="Network")
    subcategory: Optional[str] = Field(None, example="VPN")
    priority: Optional[str] = Field(None, example=None)
    intent: Optional[str] = Field(None, example=None)

class ConfidenceDetails(BaseModel):
    category_score: float = Field(..., ge=0.0, le=1.0, example=0.87)
    score_type: str = Field("estimated_probability", example="estimated_probability")
    calibrated: bool = Field(False, example=False)

class TicketClassificationResponse(BaseModel):
    ticket_id: Optional[str] = Field(None, example="INC-2001")
    classification: ClassificationDetails
    confidence: ConfidenceDetails
    review_required: bool = Field(False, example=False)
    review_reason: Optional[str] = Field(None, example=None)
    model_version: str = Field(..., example="category-tfidf-logreg-v1")
    taxonomy_version: str = Field(..., example="v1.0")
    inference_latency_ms: Optional[float] = Field(None, example=12.4)

class HealthResponse(BaseModel):
    status: str = Field("UP", example="UP")
    service: str = Field("ml-ticket-classifier", example="ml-ticket-classifier")
    model_loaded: bool = Field(..., example=True)
    model_version: str = Field(..., example="category-tfidf-logreg-v1")
    taxonomy_version: str = Field(..., example="v1.0")
    supported_categories: List[str]
    retrieval_ready: bool = Field(True, example=True)
    retrieval_index_size: int = Field(400, example=400)

# --- Phase 3: Hybrid Retrieval & Duplicate Candidate Schemas ---

class RetrievalRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, example="INC-2001")
    title: str = Field(..., min_length=1, max_length=300, example="VPN disconnects after login")
    description: str = Field(..., min_length=1, max_length=10000, example="Corporate VPN connection drops immediately after authentication on gateway 10.240.0.1.")
    affected_service: Optional[str] = Field(None, max_length=200, example="Corporate VPN")
    environment: Optional[str] = Field(None, max_length=100, example="Production")
    top_k: int = Field(5, ge=1, le=20, example=5)
    filters: Optional[Dict[str, Any]] = Field(None, example={"environment": "Production"})
    method: str = Field("hybrid", example="hybrid")

class RetrievedTicketItem(BaseModel):
    ticket_id: str = Field(..., example="INC-1001")
    title: str = Field(..., example="VPN session disconnects after authentication")
    description_excerpt: str = Field(..., example="User connects to GlobalProtect VPN using MFA...")
    affected_service: Optional[str] = Field(None, example="Corporate VPN")
    category: Optional[str] = Field(None, example="Network")
    subcategory: Optional[str] = Field(None, example="VPN")
    status: Optional[str] = Field(None, example="Resolved")
    environment: Optional[str] = Field(None, example="Production")
    data_access_level: str = Field("internal", example="internal")
    similarity_score: float = Field(..., example=0.86)
    bm25_score: float = Field(..., example=8.4)
    rrf_score: float = Field(..., example=0.032)
    hybrid_rank: int = Field(..., example=1)
    potential_duplicate: bool = Field(..., example=True)
    duplicate_confidence: float = Field(..., example=0.88)
    duplicate_reason: Optional[str] = Field(None, example="Similar VPN authentication symptoms and affected service")
    match_reason: str = Field(..., example="Identical service 'Corporate VPN' | Strong semantic symptom alignment (86%)")
    review_required: bool = Field(False, example=False)

class RetrievalResponse(BaseModel):
    query_ticket_id: Optional[str] = Field(None, example="INC-2001")
    retrieval_method: str = Field(..., example="hybrid")
    results: List[RetrievedTicketItem]
    retrieval_model_version: str = Field("hybrid-v1", example="hybrid-v1")
    retrieval_latency_ms: float = Field(..., example=14.5)

# --- Phase 4: Evidence-Backed Resolution Recommendation Schemas ---

class RecommendationRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, example="INC-2001")
    title: str = Field(..., min_length=1, max_length=300, example="VPN disconnects after login")
    description: str = Field(..., min_length=1, max_length=10000, example="Corporate VPN connection drops immediately after authentication on gateway 10.240.0.1.")
    affected_service: Optional[str] = Field(None, max_length=200, example="Corporate VPN")
    environment: Optional[str] = Field(None, max_length=100, example="Production")
    top_k: int = Field(5, ge=1, le=20, example=5)
    generation_mode: str = Field("deterministic", example="deterministic")  # "deterministic" or "rag"
    user_data_access_level: str = Field("internal", example="internal")

class EvidenceItem(BaseModel):
    source_type: str = Field("historical_ticket", example="historical_ticket")
    source_id: str = Field(..., example="INC-1001")
    title: Optional[str] = Field(None, example="VPN connection dropped")
    excerpt: str = Field(..., example="Support team verified the VPN client configuration and corrected an outdated connection profile.")
    relevance_reason: str = Field(..., example="Both tickets describe VPN disconnection immediately after authentication on Corporate VPN.")
    affected_service: Optional[str] = Field(None, example="Corporate VPN")
    environment: Optional[str] = Field(None, example="Production")
    similarity_score: Optional[float] = Field(None, example=0.88)
    status: Optional[str] = Field("Resolved", example="Resolved")

class RecommendationItem(BaseModel):
    recommendation_id: str = Field(..., example="REC-001")
    title: str = Field(..., example="Review and reapply approved VPN client connection profile")
    steps: List[str] = Field(..., example=["Verify the VPN client configuration against the approved GlobalProtect settings."])
    evidence: List[EvidenceItem]
    limitations: List[str] = Field(default_factory=list, example=["The underlying root cause has not been independently verified."])
    security_warnings: List[str] = Field(default_factory=list, example=[])

class RecommendationConfidence(BaseModel):
    level: str = Field(..., example="medium")  # high, medium, low, insufficient_evidence
    basis: str = Field(..., example="Relevant historical resolution with matching affected service; root cause equivalence unconfirmed.")

class RecommendationResponse(BaseModel):
    ticket_id: Optional[str] = Field(None, example="INC-2001")
    recommendation_status: str = Field(..., example="suggested")  # "suggested", "human_review_required", "insufficient_evidence"
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    confidence: RecommendationConfidence
    review_required: bool = Field(True, example=True)
    review_reason: Optional[str] = Field(None, example=None)
    clarification_questions: List[str] = Field(default_factory=list)
    generation_mode: str = Field("deterministic", example="deterministic")
    grounding_verified: bool = Field(True, example=True)
    model_version: str = Field("recommendation-v1", example="recommendation-v1")
    inference_latency_ms: float = Field(..., example=24.5)

