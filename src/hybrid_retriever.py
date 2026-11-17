"""
hybrid_retriever.py
===================
Hybrid Information Retrieval Engine with Reciprocal Rank Fusion (RRF),
Metadata Filtering, Access Control Enforcement, and Potential Duplicate Detection.
"""

import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ml-service.retriever")

RRF_K_DEFAULT = 60

ACCESS_HIERARCHY = {
    "public": 1,
    "internal": 2,
    "confidential": 3,
    "restricted": 4
}

def extract_technical_tokens(text: str) -> set[str]:
    """Extracts distinctive technical tokens: hex codes, IP addresses, stop codes, DLLs."""
    if not text:
        return set()
    matches = re.findall(r"0x[0-9a-fA-F]+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b[a-zA-Z0-9_\-]+\.dll\b|event id \d+", text.lower())
    return set(matches)

def evaluate_duplicate_policy(
    query_ticket: dict,
    candidate_metadata: dict,
    semantic_score: float,
    bm25_score: float
) -> tuple[bool, float, Optional[str]]:
    """
    Evaluates whether a candidate ticket is a potential duplicate incident.
    
    CRITERIA (Section 10):
    1. Strong semantic similarity (score >= 0.82)
    2. Corroborating signals:
       - Shared error code (e.g. 0x80070005, Error 800)
       - Same affected service
       - Same environment
    """
    q_title = query_ticket.get("title", "")
    q_desc = query_ticket.get("description", "")
    q_service = query_ticket.get("affected_service", "")
    q_env = query_ticket.get("environment", "")

    c_title = candidate_metadata.get("title", "")
    c_desc = candidate_metadata.get("description", "")
    c_service = candidate_metadata.get("affected_service", "")
    c_env = candidate_metadata.get("environment", "")

    # Extract technical tokens
    q_tech = extract_technical_tokens(f"{q_title} {q_desc}")
    c_tech = extract_technical_tokens(f"{c_title} {c_desc}")
    shared_tech = q_tech.intersection(c_tech)

    same_service = bool(q_service and c_service and q_service.lower() == c_service.lower())
    same_env = bool(q_env and c_env and q_env.lower() == c_env.lower())

    # Heuristic scoring
    signals = []
    confidence = 0.0

    if semantic_score >= 0.88:
        signals.append(f"Very high semantic symptom similarity ({semantic_score:.2f})")
        confidence += 0.50
    elif semantic_score >= 0.82:
        signals.append(f"High semantic symptom similarity ({semantic_score:.2f})")
        confidence += 0.35

    if shared_tech:
        signals.append(f"Identical technical identifiers: {', '.join(list(shared_tech)[:3])}")
        confidence += 0.30

    if same_service:
        signals.append(f"Matching affected service '{c_service}'")
        confidence += 0.15

    if same_env:
        signals.append(f"Same environment '{c_env}'")
        confidence += 0.05

    is_duplicate = (confidence >= 0.65) or (semantic_score >= 0.84 and (same_service or bool(shared_tech)))
    confidence = min(0.99, round(confidence, 2)) if is_duplicate else round(confidence, 2)
    reason = "; ".join(signals) if is_duplicate else None

    return (is_duplicate, confidence, reason)

def generate_match_explanation(
    query_ticket: dict,
    candidate_metadata: dict,
    semantic_score: float,
    bm25_score: float,
    rrf_rank: int
) -> str:
    """Generates an honest, grounded explanation for why the ticket matched."""
    q_text = f"{query_ticket.get('title', '')} {query_ticket.get('description', '')}"
    c_text = f"{candidate_metadata.get('title', '')} {candidate_metadata.get('description', '')}"

    q_tech = extract_technical_tokens(q_text)
    c_tech = extract_technical_tokens(c_text)
    shared_tech = q_tech.intersection(c_tech)

    reasons = []
    if shared_tech:
        reasons.append(f"Shared error identifier: {', '.join(list(shared_tech)[:2])}")

    q_service = query_ticket.get("affected_service", "")
    c_service = candidate_metadata.get("affected_service", "")
    if q_service and c_service and q_service.lower() == c_service.lower():
        reasons.append(f"Identical service '{c_service}'")

    if semantic_score >= 0.80:
        reasons.append(f"Strong semantic symptom alignment ({int(semantic_score*100)}%)")
    elif bm25_score > 0:
        reasons.append("Exact keyword phrase overlap")

    if not reasons:
        reasons.append("General domain similarity")

    return " | ".join(reasons)

class HybridRetriever:
    def __init__(self, faiss_mgr, bm25_mgr, metadata_catalog: dict, rrf_k: int = RRF_K_DEFAULT):
        self.faiss_mgr = faiss_mgr
        self.bm25_mgr = bm25_mgr
        self.metadata_catalog = metadata_catalog  # ticket_id -> metadata dict
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query_text: str,
        query_vector,
        query_metadata: dict,
        top_k: int = 5,
        filters: Optional[dict] = None,
        method: str = "hybrid"
    ) -> list[dict]:
        """
        Executes hybrid retrieval:
        1. Over-fetches candidates from FAISS (top 50) and BM25 (top 50).
        2. Applies Reciprocal Rank Fusion: RRF = sum(1 / (k + rank)).
        3. Enforces metadata filters and data_access_level constraints.
        4. Evaluates duplicate detection policy.
        5. Returns top_k results.
        """
        pool_size = max(50, top_k * 5)

        # 1. Candidate Retrieval
        faiss_candidates = {}
        if method in ["hybrid", "semantic_only"] and query_vector is not None:
            raw_faiss = self.faiss_mgr.search(query_vector, top_k=pool_size)
            for tid, score, rank in raw_faiss:
                faiss_candidates[tid] = {"score": score, "rank": rank}

        bm25_candidates = {}
        if method in ["hybrid", "bm25_only"]:
            raw_bm25 = self.bm25_mgr.search(query_text, top_k=pool_size)
            for tid, score, rank in raw_bm25:
                bm25_candidates[tid] = {"score": score, "rank": rank}

        # 2. Combine candidate IDs
        all_candidate_ids = set(faiss_candidates.keys()).union(set(bm25_candidates.keys()))

        # 3. Reciprocal Rank Fusion & Candidate Scoring
        scored_candidates = []
        user_access = (filters or {}).get("user_data_access_level", "restricted")  # default permits internal/confidential unless restricted
        user_level = ACCESS_HIERARCHY.get(user_access.lower(), 4)

        for tid in all_candidate_ids:
            meta = self.metadata_catalog.get(tid, {})

            # Exclude self if query is an existing ticket
            if query_metadata.get("ticket_id") and query_metadata["ticket_id"] == tid:
                continue

            # Check Access Level
            ticket_access = meta.get("data_access_level", "internal").lower()
            ticket_level = ACCESS_HIERARCHY.get(ticket_access, 2)
            if ticket_level > user_level:
                continue  # User unauthorized to view this ticket

            # Apply Metadata Filters
            if filters:
                if filters.get("category") and meta.get("category") != filters["category"]:
                    continue
                if filters.get("affected_service") and meta.get("affected_service") != filters["affected_service"]:
                    continue
                if filters.get("environment") and meta.get("environment") != filters["environment"]:
                    continue
                if filters.get("status") and meta.get("status") != filters["status"]:
                    continue

            # Calculate RRF Score
            rrf_score = 0.0
            sem_score = 0.0
            sem_rank = None
            bm_score = 0.0
            bm_rank = None

            if tid in faiss_candidates:
                sem_score = faiss_candidates[tid]["score"]
                sem_rank = faiss_candidates[tid]["rank"]
                if method != "bm25_only":
                    rrf_score += 1.0 / (self.rrf_k + sem_rank)

            if tid in bm25_candidates:
                bm_score = bm25_candidates[tid]["score"]
                bm_rank = bm25_candidates[tid]["rank"]
                if method != "semantic_only":
                    rrf_score += 1.0 / (self.rrf_k + bm_rank)

            scored_candidates.append({
                "ticket_id": tid,
                "metadata": meta,
                "semantic_score": round(sem_score, 4),
                "semantic_rank": sem_rank,
                "bm25_score": round(bm_score, 4),
                "bm25_rank": bm_rank,
                "rrf_score": round(rrf_score, 6)
            })

        # 4. Sort by RRF score (or method specific score)
        if method == "semantic_only":
            scored_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
        elif method == "bm25_only":
            scored_candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        else:
            scored_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 5. Build Top-K Output
        results = []
        for hybrid_rank, cand in enumerate(scored_candidates[:top_k], start=1):
            meta = cand["metadata"]
            is_dup, dup_conf, dup_reason = evaluate_duplicate_policy(
                query_ticket=query_metadata,
                candidate_metadata=meta,
                semantic_score=cand["semantic_score"],
                bm25_score=cand["bm25_score"]
            )
            match_explanation = generate_match_explanation(
                query_ticket=query_metadata,
                candidate_metadata=meta,
                semantic_score=cand["semantic_score"],
                bm25_score=cand["bm25_score"],
                rrf_rank=hybrid_rank
            )

            desc = meta.get("description", "")
            excerpt = desc[:200] + ("..." if len(desc) > 200 else "")

            results.append({
                "ticket_id": cand["ticket_id"],
                "title": meta.get("title", ""),
                "description_excerpt": excerpt,
                "affected_service": meta.get("affected_service"),
                "category": meta.get("category"),
                "subcategory": meta.get("subcategory"),
                "status": meta.get("status"),
                "environment": meta.get("environment"),
                "data_access_level": meta.get("data_access_level", "internal"),
                "similarity_score": cand["semantic_score"],
                "bm25_score": cand["bm25_score"],
                "rrf_score": cand["rrf_score"],
                "hybrid_rank": hybrid_rank,
                "potential_duplicate": is_dup,
                "duplicate_confidence": dup_conf,
                "duplicate_reason": dup_reason,
                "match_reason": match_explanation,
                "review_required": is_dup
            })

        return results
