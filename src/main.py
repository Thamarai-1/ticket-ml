"""
main.py
=======
FastAPI Inference Microservice for IT Ticket Classification & Hybrid Similarity Retrieval.

Endpoints:
- GET /health: Health check, model readiness, metadata, supported categories, retrieval readiness
- POST /classify: Phase 2 category prediction, probability inference, and threshold review
- POST /retrieve: Phase 3 semantic similarity (FAISS) + keyword (BM25) hybrid retrieval with RRF,
  metadata filtering, access control, and duplicate candidate detection.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    TicketClassificationRequest,
    TicketClassificationResponse,
    ClassificationDetails,
    ConfidenceDetails,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedTicketItem,
    RecommendationRequest,
    RecommendationResponse
)
from preprocessing import combine_ticket_text
from rules import resolve_subcategory, resolve_priority_and_intent
from taxonomy import TAXONOMY_VERSION, CATEGORIES

from embedding_service import EmbeddingService, format_ticket_for_retrieval
from faiss_index import FaissIndexManager
from bm25_service import BM25Manager
from hybrid_retriever import HybridRetriever
from recommendation_engine import RecommendationEngine
from grounding_validator import GroundingValidator
from rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml-service")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "ticket_classifier_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(MODEL_DIR, "ticket_id_map.json")
BM25_PATH = os.path.join(MODEL_DIR, "bm25_index.joblib")
TICKET_META_PATH = os.path.join(MODEL_DIR, "ticket_metadata.json")

CONFIDENCE_THRESHOLD_DEFAULT = 0.60
RETRIEVAL_MODEL_VERSION = "hybrid-rrf-v1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all Phase 2 and Phase 3 models and indexes once during startup:
    1. Scikit-learn Classification Pipeline & Metadata
    2. SentenceTransformer Embedding Service
    3. FAISS Vector Search Index
    4. BM25 Keyword Search Index
    5. Ticket Metadata Catalog
    6. Hybrid Retriever
    """
    logger.info("Starting ML Service lifespan initialization...")

    # Phase 2 Classification Setup
    app.state.pipeline = None
    app.state.metadata = {}
    app.state.confidence_threshold = CONFIDENCE_THRESHOLD_DEFAULT

    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading classification pipeline from {MODEL_PATH}...")
            app.state.pipeline = joblib.load(MODEL_PATH)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    app.state.metadata = json.load(f)
                    app.state.confidence_threshold = app.state.metadata.get(
                        "confidence_threshold", CONFIDENCE_THRESHOLD_DEFAULT
                    )
            logger.info("Classification pipeline ready.")
        except Exception as e:
            logger.error(f"Failed to load classification pipeline: {e}")
    else:
        logger.warning("Classification pipeline artifact not found.")

    # Phase 3 Hybrid Retrieval Setup
    logger.info("Initializing SentenceTransformer embedding service...")
    app.state.embedding_service = EmbeddingService()

    logger.info("Loading FAISS and BM25 retrieval indexes...")
    app.state.faiss_mgr = FaissIndexManager(dim=app.state.embedding_service.dim)
    app.state.bm25_mgr = BM25Manager()
    app.state.metadata_catalog = {}

    # Load ticket metadata catalog
    if os.path.exists(TICKET_META_PATH):
        try:
            with open(TICKET_META_PATH, "r", encoding="utf-8") as f:
                app.state.metadata_catalog = json.load(f)
            logger.info(f"Loaded ticket metadata catalog ({len(app.state.metadata_catalog)} records).")
        except Exception as e:
            logger.warning(f"Could not load ticket metadata: {e}")

    # Load FAISS index
    faiss_loaded = app.state.faiss_mgr.load(FAISS_INDEX_PATH, ID_MAP_PATH)
    # Load BM25 index
    bm25_loaded = app.state.bm25_mgr.load(BM25_PATH)

    # If indexes missing, auto-build from historical_tickets_cleaned.json if present
    if not faiss_loaded or not bm25_loaded or len(app.state.metadata_catalog) == 0:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "historical_tickets_cleaned.json")
        if os.path.exists(dataset_path):
            try:
                from build_indexes import build_all_indexes
                logger.info("Building retrieval indexes from dataset on startup...")
                build_all_indexes(dataset_path, MODEL_DIR)
                app.state.faiss_mgr.load(FAISS_INDEX_PATH, ID_MAP_PATH)
                app.state.bm25_mgr.load(BM25_PATH)
                if os.path.exists(TICKET_META_PATH):
                    with open(TICKET_META_PATH, "r", encoding="utf-8") as f:
                        app.state.metadata_catalog = json.load(f)
                logger.info("Startup index build completed successfully.")
            except Exception as e:
                logger.error(f"Failed to auto-build indexes: {e}")

    app.state.hybrid_retriever = HybridRetriever(
        faiss_mgr=app.state.faiss_mgr,
        bm25_mgr=app.state.bm25_mgr,
        metadata_catalog=app.state.metadata_catalog,
        rrf_k=60
    )
    logger.info("Hybrid Retriever initialized successfully.")

    # Phase 4 Recommendation & Grounding Setup
    app.state.recommendation_engine = RecommendationEngine(app.state.metadata_catalog)
    app.state.grounding_validator = GroundingValidator(app.state.metadata_catalog)
    app.state.rag_service = RAGService(app.state.recommendation_engine, app.state.grounding_validator)
    logger.info("Recommendation Engine and RAG Service ready.")

    yield

    logger.info("Shutting down ML Service...")

app = FastAPI(
    title="IT Ticket Classification & Hybrid Retrieval Service",
    description="Microservice providing classification, FAISS semantic search, BM25 lexical search, RRF fusion, and duplicate detection.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health, model readiness, and index size."""
    model_loaded = getattr(app.state, "pipeline", None) is not None
    model_version = app.state.metadata.get("model_version", "uninitialized") if hasattr(app.state, "metadata") else "uninitialized"
    index_size = len(app.state.faiss_mgr.id_map) if hasattr(app.state, "faiss_mgr") else 0

    return HealthResponse(
        status="UP",
        service="ml-ticket-classifier",
        model_loaded=model_loaded,
        model_version=model_version,
        taxonomy_version=TAXONOMY_VERSION,
        supported_categories=CATEGORIES,
        retrieval_ready=index_size > 0,
        retrieval_index_size=index_size
    )

# --- Phase 2: Classification Endpoint ---

@app.post("/classify", response_model=TicketClassificationResponse)
async def classify_ticket(req: TicketClassificationRequest):
    t0 = time.perf_counter()

    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification model is not loaded."
        )

    pipeline = app.state.pipeline
    model_version = app.state.metadata.get("model_version", "category-tfidf-logreg-v1")
    threshold = app.state.confidence_threshold

    combined_text = combine_ticket_text(req.title, req.description, req.affected_service)

    # Empty or short input handling
    if not combined_text or len(combined_text.strip()) == 0:
        return TicketClassificationResponse(
            ticket_id=req.ticket_id,
            classification=ClassificationDetails(category=None, subcategory=None, priority=None, intent=None),
            confidence=ConfidenceDetails(category_score=0.0, score_type="estimated_probability", calibrated=False),
            review_required=True,
            review_reason="Input ticket text is empty or blank",
            model_version=model_version,
            taxonomy_version=TAXONOMY_VERSION,
            inference_latency_ms=round((time.perf_counter() - t0) * 1000, 2)
        )

    if len(combined_text.strip()) < 10:
        return TicketClassificationResponse(
            ticket_id=req.ticket_id,
            classification=ClassificationDetails(category=None, subcategory=None, priority=None, intent=None),
            confidence=ConfidenceDetails(category_score=0.0, score_type="estimated_probability", calibrated=False),
            review_required=True,
            review_reason="Input ticket text is too short (< 10 chars) for reliable inference",
            model_version=model_version,
            taxonomy_version=TAXONOMY_VERSION,
            inference_latency_ms=round((time.perf_counter() - t0) * 1000, 2)
        )

    tfidf_step = pipeline.named_steps["tfidf"]
    clf_step = pipeline.named_steps["clf"]
    vector = tfidf_step.transform([combined_text])

    if vector.nnz == 0:
        return TicketClassificationResponse(
            ticket_id=req.ticket_id,
            classification=ClassificationDetails(category=None, subcategory=None, priority=None, intent=None),
            confidence=ConfidenceDetails(category_score=0.0, score_type="estimated_probability", calibrated=False),
            review_required=True,
            review_reason="Unrecognized technical vocabulary completely outside trained taxonomy",
            model_version=model_version,
            taxonomy_version=TAXONOMY_VERSION,
            inference_latency_ms=round((time.perf_counter() - t0) * 1000, 2)
        )

    probs = clf_step.predict_proba(vector)[0]
    classes = clf_step.classes_
    top_idx = int(np.argmax(probs))

    predicted_category = str(classes[top_idx])
    category_score = float(round(probs[top_idx], 4))

    review_required = category_score < threshold
    review_reason = (
        f"Low classification confidence ({category_score:.2f} < {threshold:.2f}); ambiguous symptom patterns detected"
        if review_required else None
    )

    subcategory = resolve_subcategory(predicted_category, combined_text)
    priority, intent = resolve_priority_and_intent(combined_text)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return TicketClassificationResponse(
        ticket_id=req.ticket_id,
        classification=ClassificationDetails(
            category=predicted_category,
            subcategory=subcategory,
            priority=priority,
            intent=intent
        ),
        confidence=ConfidenceDetails(
            category_score=category_score,
            score_type="estimated_probability",
            calibrated=False
        ),
        review_required=review_required,
        review_reason=review_reason,
        model_version=model_version,
        taxonomy_version=TAXONOMY_VERSION,
        inference_latency_ms=latency_ms
    )

# --- Phase 3: Hybrid Retrieval Endpoint ---

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_similar_tickets(req: RetrievalRequest):
    """
    Hybrid Information Retrieval & Duplicate Candidate Detection Endpoint:
    1. Formats incoming ticket describing problem symptoms (omitting resolution_text).
    2. Encodes query into dense semantic vector using SentenceTransformer (dim=384).
    3. Searches FAISS semantic index & BM25 keyword index.
    4. Merges candidate rankings using Reciprocal Rank Fusion (RRF k=60).
    5. Applies metadata filters and access level restrictions.
    6. Identifies potential duplicate incidents using grounded heuristic policy.
    7. Returns top-k ranked historical matches with similarity evidence.
    """
    t0 = time.perf_counter()

    if getattr(app.state, "hybrid_retriever", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not initialized. Please ensure indexes are built."
        )

    query_dict = {
        "ticket_id": req.ticket_id,
        "title": req.title,
        "description": req.description,
        "affected_service": req.affected_service,
        "environment": req.environment
    }

    # Format problem text for embedding and retrieval
    formatted_query_text = format_ticket_for_retrieval(query_dict)

    # Encode semantic vector
    query_vector = app.state.embedding_service.encode_one(formatted_query_text)

    # Execute Hybrid Retrieval
    results = app.state.hybrid_retriever.retrieve(
        query_text=formatted_query_text,
        query_vector=query_vector,
        query_metadata=query_dict,
        top_k=req.top_k,
        filters=req.filters,
        method=req.method
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return RetrievalResponse(
        query_ticket_id=req.ticket_id,
        retrieval_method=req.method,
        results=[RetrievedTicketItem(**item) for item in results],
        retrieval_model_version=RETRIEVAL_MODEL_VERSION,
        retrieval_latency_ms=latency_ms
    )

# --- Phase 4: Evidence-Backed Resolution Recommendation Endpoint ---

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_resolution(req: RecommendationRequest):
    """
    Evidence-Backed Resolution Recommendation Endpoint:
    1. Reuses Phase 3 hybrid retrieval to search historical candidates.
    2. Filters for verified 'Resolved' / 'Closed' tickets with usable resolution_text.
    3. Normalizes steps, preserves technical tokens (IPs, error codes, commands).
    4. Detects contradictions, privileged commands, and security-sensitive tasks.
    5. Applies GroundingValidator to verify source citations and exact excerpts.
    6. Returns structured action plan with qualitative confidence and human-review flag.
    """
    t0 = time.perf_counter()

    if getattr(app.state, "rag_service", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not initialized. Please ensure models and metadata are loaded."
        )

    query_dict = {
        "ticket_id": req.ticket_id,
        "title": req.title,
        "description": req.description,
        "affected_service": req.affected_service,
        "environment": req.environment
    }

    # Format problem text strictly omitting resolution_text
    formatted_query_text = format_ticket_for_retrieval(query_dict)

    # Encode semantic vector
    query_vector = app.state.embedding_service.encode_one(formatted_query_text)

    # Retrieve candidate historical tickets (fetching top 10 candidates for robust evidence pool)
    candidates = app.state.hybrid_retriever.retrieve(
        query_text=formatted_query_text,
        query_vector=query_vector,
        query_metadata=query_dict,
        top_k=max(10, req.top_k * 2),
        filters={"user_data_access_level": req.user_data_access_level},
        method="hybrid"
    )

    # Generate grounded resolution recommendations
    response = app.state.rag_service.generate(
        query_context=query_dict,
        retrieved_items=candidates,
        top_k=req.top_k,
        user_access_level=req.user_data_access_level,
        force_mode=req.generation_mode
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.inference_latency_ms = latency_ms

    return response
