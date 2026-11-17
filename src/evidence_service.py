"""
evidence_service.py
===================
Evidence Eligibility, Relevance Ranking, Contradiction Detection, and Step Extraction.

Rules:
1. Only Resolved/Closed tickets with usable resolution_text (>= 10 chars) are eligible.
2. Unresolved tickets (New, In Progress, Pending, Cancelled) are strictly excluded from resolution evidence.
3. Access level hierarchy (public < internal < confidential < restricted) is strictly enforced.
4. Preserves technical tokens and exact excerpts without altering technical meaning.
"""

import re
from typing import Dict, Any, List, Tuple, Optional

ACCESS_HIERARCHY = {
    "public": 1,
    "internal": 2,
    "confidential": 3,
    "restricted": 4
}

PRIVILEGED_KEYWORDS = [
    "sudo", "root access", "admin rights", "domain admin", "active directory admin",
    "modify firewall", "disable firewall", "drop table", "truncate table",
    "format c:", "rm -rf", "delete schema", "privileged credentials", "registry modification"
]

CONTRADICTORY_PAIRS = [
    ("upgrade", "downgrade"),
    ("enable", "disable"),
    ("increase", "decrease"),
    ("add", "remove"),
    ("install", "uninstall"),
    ("whitelist", "blacklist"),
    ("block", "allow")
]

PLACEHOLDER_RESOLUTIONS = {
    "n/a", "na", "fixed", "resolved", "done", "none", "closed", "ok", "completed", "test", "solved"
}

def is_evidence_eligible(
    candidate: Dict[str, Any],
    query_context: Dict[str, Any],
    user_access_level: str = "internal"
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether a candidate historical ticket is eligible to serve as
    factual resolution evidence.
    """
    cand_id = candidate.get("ticket_id")
    query_id = query_context.get("ticket_id")

    # Exclude self-matches
    if cand_id and query_id and cand_id.strip().upper() == query_id.strip().upper():
        return False, "Candidate is identical to the query ticket"

    # Rule 1: Verified state check - Must be Resolved or Closed
    status = candidate.get("status", "")
    if status not in ["Resolved", "Closed"]:
        return False, f"Ticket {cand_id} status is '{status}'; only 'Resolved' or 'Closed' tickets can serve as resolution evidence"

    # Rule 2: Resolution text existence and usability
    res_text = candidate.get("resolution_text")
    if not res_text or not isinstance(res_text, str):
        return False, f"Ticket {cand_id} has no resolution_text recorded"

    cleaned_res = res_text.strip()
    if len(cleaned_res) < 10 or cleaned_res.lower() in PLACEHOLDER_RESOLUTIONS:
        return False, f"Ticket {cand_id} resolution_text is empty, incomplete, or a placeholder ('{cleaned_res}')"

    # Rule 3: Data Access Level Authorization
    cand_access = candidate.get("data_access_level", "internal").lower()
    user_access = user_access_level.lower()

    cand_level_num = ACCESS_HIERARCHY.get(cand_access, 2)
    user_level_num = ACCESS_HIERARCHY.get(user_access, 2)

    if cand_level_num > user_level_num:
        return False, f"Access restricted: ticket access level '{cand_access}' exceeds user clearance '{user_access}'"

    return True, None


def score_evidence_relevance(
    candidate: Dict[str, Any],
    query_context: Dict[str, Any],
    semantic_score: float,
    bm25_score: float
) -> float:
    """
    Calculates composite evidence relevance score combining semantic similarity,
    lexical match, affected service alignment, and environment compatibility.
    """
    # Normalized BM25 score (bounded 0.0 to 1.0, typical max BM25 is ~20)
    norm_bm25 = min(1.0, max(0.0, bm25_score / 20.0))

    # Base similarity weight: 50% semantic, 20% BM25
    score = (0.50 * max(0.0, semantic_score)) + (0.20 * norm_bm25)

    # Service compatibility (+0.15 for exact match)
    q_service = (query_context.get("affected_service") or "").strip().lower()
    c_service = (candidate.get("affected_service") or "").strip().lower()
    if q_service and c_service and q_service == c_service:
        score += 0.15
    elif q_service and c_service and (q_service in c_service or c_service in q_service):
        score += 0.08

    # Environment compatibility (+0.10 for exact match)
    q_env = (query_context.get("environment") or "").strip().lower()
    c_env = (candidate.get("environment") or "").strip().lower()
    if q_env and c_env and q_env == c_env:
        score += 0.10

    # Verification bonus (+0.05 for Verified / Closed historical tickets)
    if candidate.get("status") == "Closed":
        score += 0.05

    return round(min(1.0, score), 4)


def extract_resolution_steps(resolution_text: str) -> List[str]:
    """
    Normalizes resolution_text and segments it into clean, actionable steps
    while strictly retaining all technical tokens, paths, error codes, and identifiers.
    """
    if not resolution_text:
        return []

    # Clean redundant whitespace
    text = re.sub(r'\s+', ' ', resolution_text.strip())

    # Split by periods, semicolons, or numbered list indicators
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\d[.)])\s+', text)

    steps = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Remove trailing period if standalone
        if len(s_clean) >= 8:
            steps.append(s_clean)

    # If sentence splitting yielded nothing, return full normalized text
    if not steps and len(text) >= 8:
        steps.append(text)

    return steps


def detect_security_or_privileged_actions(steps: List[str]) -> Tuple[bool, List[str]]:
    """
    Scans resolution steps for privileged operations, credential handling, or security-sensitive tasks.
    """
    found_warnings = []
    combined = " ".join(steps).lower()

    for kw in PRIVILEGED_KEYWORDS:
        if kw in combined:
            found_warnings.append(f"Requires elevated privilege or security authorization: '{kw}'")

    has_warning = len(found_warnings) > 0
    return has_warning, found_warnings


def detect_evidence_contradictions(eligible_candidates: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Checks if multiple eligible tickets propose contradictory interventions
    (e.g., Ticket A upgraded package, Ticket B downgraded package; or enable vs disable).
    """
    if len(eligible_candidates) < 2:
        return False, []

    conflicts = []
    # Inspect pairwise resolutions
    for i in range(len(eligible_candidates)):
        res_i = (eligible_candidates[i].get("resolution_text") or "").lower()
        id_i = eligible_candidates[i].get("ticket_id")

        for j in range(i + 1, len(eligible_candidates)):
            res_j = (eligible_candidates[j].get("resolution_text") or "").lower()
            id_j = eligible_candidates[j].get("ticket_id")

            for word_a, word_b in CONTRADICTORY_PAIRS:
                if (word_a in res_i and word_b in res_j) or (word_b in res_i and word_a in res_j):
                    conflicts.append(
                        f"Contradictory actions detected between {id_i} ('{word_a}') and {id_j} ('{word_b}'). Human review required to determine applicable procedure."
                    )

    has_conflict = len(conflicts) > 0
    return has_conflict, conflicts
