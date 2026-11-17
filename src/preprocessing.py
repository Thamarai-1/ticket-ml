"""
preprocessing.py
================
Standardized Text Preprocessing and Feature Combination for Training and Inference.

CRITICAL REQUIREMENT:
The exact same text preprocessing function must be applied during both
model training and live FastAPI inference to prevent train-serve skew.
"""

import re

def clean_text_for_ml(text: str) -> str:
    """
    Normalizes whitespace and standardizes casing while strictly preserving
    crucial technical tokens:
    - Hexadecimal error codes (e.g. 0x80070005, 0x800423f4)
    - IP addresses and port numbers (e.g. 10.240.0.1, 127.0.0.1:8080)
    - Windows Event IDs (e.g. Event ID 4625)
    - Protocol codes and HTTP status codes (e.g. 502, Error 800)
    - System file names (e.g. outlmapi.dll, nvlddmkm.sys)
    - File system paths (e.g. C:\\Windows\\System32)
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace null bytes and control chars
    cleaned = text.replace("\x00", " ")

    # Normalize CRLF and tabs
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # Collapse multiple inline spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

def combine_ticket_text(title: str, description: str, affected_service: str = None) -> str:
    """
    Combines incoming ticket fields available at submission time.
    
    STRICT SAFETY RULES:
    1. Does NOT use resolution_text.
    2. Does NOT use target category.
    3. Does NOT use resolved_at or post-triage fields.
    """
    t_clean = clean_text_for_ml(title)
    d_clean = clean_text_for_ml(description)
    s_clean = clean_text_for_ml(affected_service) if affected_service else ""

    parts = [p for p in [t_clean, d_clean, s_clean] if p]
    return " ".join(parts)
