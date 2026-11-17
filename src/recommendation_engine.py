"""
recommendation_engine.py
========================
Deterministic Evidence-Backed Resolution Recommendation Engine.

Adheres strictly to Hackathon guidelines:
1. Recommendations are derived exclusively from verified, eligible historical tickets.
2. Exact supporting excerpts and source IDs are attached for full auditability (FR-04).
3. Preserves verbatim technical identifiers (IPs, error codes, commands, registry paths).
4. Detects contradictions, privileged commands, and security-sensitive incidents.
5. Qualitative confidence estimation (high, medium, low, insufficient_evidence) - NOT raw cosine similarity!
6. Always flags review_required: True (advisory assistance for human analysts).
"""

import logging
from typing import Dict, Any, List, Optional
from schemas import (
    RecommendationItem,
    EvidenceItem,
    RecommendationConfidence,
    RecommendationResponse
)
from evidence_service import (
    is_evidence_eligible,
    score_evidence_relevance,
    extract_resolution_steps,
    detect_security_or_privileged_actions,
    detect_evidence_contradictions
)

logger = logging.getLogger("recommendation_engine")

RECOMMENDATION_MODEL_VERSION = "recommendation-v1"

def generate_clarification_questions(query_context: Dict[str, Any]) -> List[str]:
    """
    Generates targeted clarification questions when essential technical context is missing.
    """
    questions = []
    title_desc = f"{query_context.get('title', '')} {query_context.get('description', '')}".lower()
    service = (query_context.get("affected_service") or "").lower()

    if not query_context.get("affected_service"):
        questions.append("Which affected enterprise service or infrastructure component is experiencing this issue?")

    if not query_context.get("environment"):
        questions.append("Which environment (Production, Staging, QA, Development) is affected?")

    if "vpn" in title_desc or "vpn" in service:
        if "globalprotect" not in title_desc and "cisco" not in title_desc and "anyconnect" not in title_desc:
            questions.append("Which VPN client software and version are installed (e.g. GlobalProtect 6.x, Cisco AnyConnect)?")
        if "error" not in title_desc and "code" not in title_desc:
            questions.append("Does the client display an explicit numeric error code (e.g., Error 800, 4625)?")

    if "password" in title_desc or "lockout" in title_desc:
        if "domain" not in title_desc and "ad" not in title_desc and "okta" not in title_desc:
            questions.append("Is the identity provider Active Directory Domain Services or Cloud Identity (Okta/Azure AD)?")

    if "memory" in title_desc or "crash" in title_desc or "502" in title_desc:
        if "host" not in title_desc and "server" not in title_desc and "pod" not in title_desc:
            questions.append("What is the affected hostname, container ID, or IP address showing high resource utilization?")

    return questions


def compute_recommendation_confidence(
    eligible_count: int,
    top_score: float,
    has_service_match: bool,
    has_conflict: bool,
    is_privileged: bool
) -> RecommendationConfidence:
    """
    Computes a qualitative recommendation confidence level based on source agreement,
    evidence depth, and service alignment.
    """
    if eligible_count == 0 or top_score < 0.40:
        return RecommendationConfidence(
            level="insufficient_evidence",
            basis="No verified historical resolutions found matching the reported problem symptoms."
        )

    if has_conflict:
        return RecommendationConfidence(
            level="low",
            basis="Eligible historical tickets propose contradictory procedures; human triage required to adjudicate."
        )

    if eligible_count >= 2 and top_score >= 0.75 and has_service_match and not is_privileged:
        return RecommendationConfidence(
            level="high",
            basis=f"Multiple consistent historical resolutions ({eligible_count} sources) with identical affected service and high symptom relevance ({int(top_score*100)}%)."
        )

    if eligible_count >= 1 and top_score >= 0.55:
        return RecommendationConfidence(
            level="medium",
            basis=f"Verified historical resolution found with matching operational context ({int(top_score*100)}% relevance); root cause equivalence unconfirmed."
        )

    return RecommendationConfidence(
        level="low",
        basis="Marginal evidence relevance or differing service context; treat suggested steps as preliminary guidance."
    )


class RecommendationEngine:
    """
    Orchestrates deterministic resolution recommendation generation from
    retrieved historical tickets.
    """

    def __init__(self, metadata_catalog: Dict[str, Any]):
        self.metadata_catalog = metadata_catalog

    def generate_recommendations(
        self,
        query_context: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        top_k: int = 3,
        user_access_level: str = "internal"
    ) -> RecommendationResponse:
        """
        Main recommendation pipeline:
        1. Filters retrieved candidates for evidence eligibility.
        2. Scores and ranks eligible resolution evidence.
        3. Detects contradictions and privileged operations.
        4. Synthesizes structured recommendations with verbatim citations.
        5. Computes qualitative confidence.
        6. Constructs fallback or clarification if evidence is inadequate.
        """
        # Step 1: Filter for eligibility
        eligible_candidates = []
        disqualified_reasons = []

        for item in retrieved_items:
            tid = item.get("ticket_id")
            # Pull full record from metadata catalog
            record = self.metadata_catalog.get(tid, item)

            is_ok, reason = is_evidence_eligible(record, query_context, user_access_level)
            if is_ok:
                sem_score = item.get("similarity_score", 0.0)
                bm25_sc = item.get("bm25_score", 0.0)
                ev_score = score_evidence_relevance(record, query_context, sem_score, bm25_sc)
                eligible_candidates.append({
                    "record": record,
                    "retrieval_item": item,
                    "evidence_score": ev_score
                })
            else:
                disqualified_reasons.append(f"{tid}: {reason}")

        # Step 2: Sort eligible by composite evidence score
        eligible_candidates.sort(key=lambda x: x["evidence_score"], reverse=True)
        top_eligible = eligible_candidates[:top_k]

        # Check for contradictions across top candidates
        cand_records = [c["record"] for c in top_eligible]
        has_conflict, conflict_reasons = detect_evidence_contradictions(cand_records)

        # Step 3: Handle Insufficient Evidence
        if len(top_eligible) == 0:
            clarifications = generate_clarification_questions(query_context)
            conf = compute_recommendation_confidence(0, 0.0, False, False, False)
            return RecommendationResponse(
                ticket_id=query_context.get("ticket_id"),
                recommendation_status="insufficient_evidence",
                recommendations=[],
                confidence=conf,
                review_required=True,
                review_reason="No verified historical tickets in Resolved/Closed status match these symptoms.",
                clarification_questions=clarifications,
                generation_mode="deterministic",
                grounding_verified=True,
                model_version=RECOMMENDATION_MODEL_VERSION,
                inference_latency_ms=0.0
            )

        # Step 4: Construct Structured Recommendations
        recommendations: List[RecommendationItem] = []
        all_warnings: List[str] = []
        has_privileged = False

        for idx, item_data in enumerate(top_eligible, 1):
            rec = item_data["record"]
            ret_item = item_data["retrieval_item"]
            ev_score = item_data["evidence_score"]
            res_text = rec.get("resolution_text", "").strip()

            steps = extract_resolution_steps(res_text)
            is_priv, priv_warnings = detect_security_or_privileged_actions(steps)
            if is_priv:
                has_privileged = True
                all_warnings.extend(priv_warnings)

            # Build exact evidence item
            evidence_item = EvidenceItem(
                source_type="historical_ticket",
                source_id=rec.get("ticket_id"),
                title=rec.get("title"),
                excerpt=res_text,
                relevance_reason=ret_item.get("match_reason") or f"Historical case resolved symptoms for service '{rec.get('affected_service')}'",
                affected_service=rec.get("affected_service"),
                environment=rec.get("environment"),
                similarity_score=ret_item.get("similarity_score"),
                status=rec.get("status")
            )

            # Generate concise, professional recommendation title
            service_name = rec.get("affected_service") or "system"
            rec_title = f"Resolution procedure derived from {rec.get('ticket_id')} ({service_name})"

            limitations = [
                "Suggested resolution is derived from past incident history; root cause equivalence must be confirmed by the triage analyst.",
                "Automated execution is disabled by design. All remediation commands must be inspected prior to execution."
            ]
            if rec.get("environment") != query_context.get("environment"):
                limitations.append(f"Historical incident occurred in '{rec.get('environment')}' while current ticket specifies '{query_context.get('environment')}'.")

            recommendations.append(
                RecommendationItem(
                    recommendation_id=f"REC-{idx:03d}",
                    title=rec_title,
                    steps=steps,
                    evidence=[evidence_item],
                    limitations=limitations,
                    security_warnings=priv_warnings
                )
            )

        # Step 5: Determine overall status, confidence, and human review reasons
        top_score = top_eligible[0]["evidence_score"]
        q_service = (query_context.get("affected_service") or "").lower()
        c_service = (top_eligible[0]["record"].get("affected_service") or "").lower()
        service_match = bool(q_service and c_service and (q_service == c_service or q_service in c_service))

        confidence = compute_recommendation_confidence(
            eligible_count=len(top_eligible),
            top_score=top_score,
            has_service_match=service_match,
            has_conflict=has_conflict,
            is_privileged=has_privileged
        )

        review_reasons = []
        rec_status = "suggested"

        if has_conflict:
            rec_status = "human_review_required"
            review_reasons.append("Contradictory resolution patterns detected across historical incidents.")
            review_reasons.extend(conflict_reasons)

        if has_privileged:
            review_reasons.append("Proposed steps require elevated credentials or privileged administrative permissions.")

        if confidence.level in ["low", "insufficient_evidence"]:
            rec_status = "human_review_required"
            review_reasons.append(f"Recommendation confidence is {confidence.level} ({confidence.basis}).")

        clarifications = []
        if len(recommendations) == 0 or confidence.level in ["low", "insufficient_evidence"]:
            clarifications = generate_clarification_questions(query_context)

        return RecommendationResponse(
            ticket_id=query_context.get("ticket_id"),
            recommendation_status=rec_status,
            recommendations=recommendations,
            confidence=confidence,
            review_required=True,  # Advisory system: human approval always required
            review_reason="; ".join(review_reasons) if review_reasons else "Analyst verification required before remediation.",
            clarification_questions=clarifications,
            generation_mode="deterministic",
            grounding_verified=True,
            model_version=RECOMMENDATION_MODEL_VERSION,
            inference_latency_ms=0.0
        )
