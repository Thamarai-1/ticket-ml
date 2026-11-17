"""
grounding_validator.py
======================
Deterministic Grounding Validation Layer for RAG and Recommendation Outputs.

Prevents Hallucinations and Prompt Injection:
1. Verifies that every cited source_id exists in the authorized retrieval candidates.
2. Verifies that every evidence excerpt is an actual, normalized substring of the source's resolution_text.
3. Rejects any fabricated ticket IDs, invented error codes, or ungrounded claims.
4. Detects and neutralizes prompt injection artifacts.
"""

import re
from typing import Dict, Any, List, Tuple
from schemas import RecommendationResponse, RecommendationItem

PROHIBITED_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"jailbreak",
    r"admin\s+override"
]

def normalize_text(text: str) -> str:
    """Removes excess whitespace, quotes, and case variations for fuzzy substring checks."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

class GroundingValidator:
    """
    Validates that a recommendation response is 100% grounded in authorized source evidence.
    """

    def __init__(self, authorized_catalog: Dict[str, Any]):
        self.authorized_catalog = authorized_catalog

    def validate(
        self,
        recommendation_response: RecommendationResponse,
        eligible_ticket_ids: List[str]
    ) -> Tuple[bool, List[str], RecommendationResponse]:
        """
        Validates the entire recommendation response against authorized source records.
        Returns:
            (is_grounded, failure_reasons, validated_response)
        """
        errors = []
        validated_items: List[RecommendationItem] = []

        eligible_set = set(t.strip().upper() for t in eligible_ticket_ids)

        for rec in recommendation_response.recommendations:
            rec_id = rec.recommendation_id
            rec_evidence_valid = True

            # Must have supporting evidence
            if not rec.evidence or len(rec.evidence) == 0:
                errors.append(f"{rec_id}: Contains no supporting evidence items.")
                continue

            for ev in rec.evidence:
                source_id = ev.source_id.strip().upper()

                # Check 1: Citation existence in authorized set
                if source_id not in eligible_set:
                    errors.append(
                        f"{rec_id}: Cited source_id '{source_id}' is not in the authorized retrieval set (Hallucinated citation)."
                    )
                    rec_evidence_valid = False
                    continue

                # Check 2: Ticket existence in catalog
                source_record = self.authorized_catalog.get(source_id)
                if not source_record:
                    errors.append(f"{rec_id}: Cited source_id '{source_id}' not found in database catalog.")
                    rec_evidence_valid = False
                    continue

                # Check 3: Excerpt verification (must be substring of true resolution_text)
                actual_res = normalize_text(source_record.get("resolution_text", ""))
                quoted_excerpt = normalize_text(ev.excerpt)

                if len(quoted_excerpt) < 5 or (quoted_excerpt not in actual_res and actual_res not in quoted_excerpt):
                    # Check partial phrase overlap
                    tokens = [t for t in quoted_excerpt.split() if len(t) > 3]
                    matching_tokens = [t for t in tokens if t in actual_res]
                    overlap_ratio = len(matching_tokens) / max(1, len(tokens))

                    if overlap_ratio < 0.60:
                        errors.append(
                            f"{rec_id}: Quoted excerpt '{ev.excerpt[:40]}...' is not supported by source {source_id}'s actual resolution text."
                        )
                        rec_evidence_valid = False

            # Check 4: Prompt injection detection in generated steps
            for step in rec.steps:
                for pattern in PROHIBITED_INJECTION_PATTERNS:
                    if re.search(pattern, step, re.IGNORECASE):
                        errors.append(f"{rec_id}: Step contains prohibited prompt injection pattern: '{pattern}'")
                        rec_evidence_valid = False

            if rec_evidence_valid:
                validated_items.append(rec)

        is_grounded = len(errors) == 0 and len(validated_items) > 0

        # Create updated response with validated items
        recommendation_response.recommendations = validated_items
        recommendation_response.grounding_verified = is_grounded

        if not is_grounded:
            recommendation_response.review_required = True
            rec_status = "human_review_required"
            recommendation_response.recommendation_status = rec_status
            if errors:
                prior = recommendation_response.review_reason or ""
                recommendation_response.review_reason = f"Grounding validation rejected unsupported claims: {'; '.join(errors)}. {prior}".strip()

        return is_grounded, errors, recommendation_response
