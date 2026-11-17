"""
ingest_hf_dataset.py
====================
Ingestion and Normalization Pipeline for Hugging Face Dataset:
'Tobi-Bueck/customer-support-tickets' (61,765 tickets)

Maps raw tickets to the standardized IT Ticket Assistant schema:
- ticket_id: INC-XXXX
- title: subject
- description: body
- resolution_text: answer
- category: mapped to core IT taxonomy (Network, Access & Identity, Hardware, Software, Server & Infrastructure, Email & Collaboration)
- priority: critical (P1), high (P2), medium (P3), low (P4)
- intent: Incident, Service Request, Problem, Change Request
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import re
import argparse
from datetime import datetime, timezone
from datasets import load_dataset
from taxonomy import CATEGORIES
from dataset_prep import prepare_splits

CATEGORY_KEYWORDS = {
    "Network": [
        "network", "vpn", "wifi", "wi-fi", "dns", "dhcp", "ip address", "firewall",
        "bandwidth", "router", "switch", "latency", "packet loss", "connectivity", "outage", "disruption"
    ],
    "Access & Identity": [
        "account", "password", "login", "logon", "sso", "mfa", "2fa", "active directory",
        "permission", "access", "ldap", "credential", "unlock", "reset", "auth"
    ],
    "Hardware": [
        "printer", "scanner", "monitor", "screen", "keyboard", "mouse", "laptop", "desktop",
        "projector", "cable", "dock", "hard drive", "ssd", "ram", "power supply", "device"
    ],
    "Software": [
        "app", "software", "install", "update", "patch", "crash", "error code", "bug",
        "license", "browser", "chrome", "firefox", "word", "excel", "application"
    ],
    "Server & Infrastructure": [
        "server", "cloud", "aws", "azure", "linux", "windows server", "database", "sql",
        "vm", "virtual machine", "container", "docker", "kubernetes", "cpu", "storage", "cluster"
    ],
    "Email & Collaboration": [
        "email", "outlook", "exchange", "mailbox", "spam", "phishing", "teams", "slack",
        "zoom", "calendar", "meeting", "smtp", "imap"
    ]
}

def map_category(subject: str, body: str, queue: str, tags: list) -> str:
    combined = f"{subject} {body} {queue} {' '.join(tags)}".lower()

    # Score each category based on keyword hits
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', combined))
        scores[cat] = score

    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat

    # Queue-based heuristics
    q_lower = (queue or "").lower()
    if "technical" in q_lower or "it" in q_lower:
        return "Software"
    if "service outage" in q_lower or "maintenance" in q_lower:
        return "Network"
    if "billing" in q_lower:
        return "Access & Identity"
    return "Software"

def map_priority(raw_priority: str) -> str:
    p = (raw_priority or "").lower()
    if p == "critical":
        return "P1"
    if p == "high":
        return "P2"
    if p in ("medium", "normal"):
        return "P3"
    return "P4"

def map_intent(raw_type: str) -> str:
    t = (raw_type or "").lower()
    if "incident" in t:
        return "Incident"
    if "request" in t:
        return "Service Request"
    if "problem" in t:
        return "Problem"
    if "change" in t:
        return "Change Request"
    return "Incident"

def ingest_dataset(lang: str = "en", limit: int = 5000, start_id: int = 10001):
    print(f"[*] Loading 'Tobi-Bueck/customer-support-tickets' from Hugging Face...")
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")

    records = []
    current_id = start_id

    for r in ds:
        if lang != "all" and r.get("language") != lang:
            continue

        subject = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        answer = (r.get("answer") or "").strip()

        if len(subject) < 5 or len(body) < 10:
            continue

        tags = [r.get(f"tag_{i}") for i in range(1, 9) if r.get(f"tag_{i}")]
        queue = r.get("queue") or "Technical Support"

        category = map_category(subject, body, queue, tags)
        priority = map_priority(r.get("priority"))
        intent = map_intent(r.get("type"))

        ticket_id = f"INC-{current_id}"
        current_id += 1

        record = {
            "ticket_id": ticket_id,
            "title": subject,
            "description": body,
            "affected_service": queue,
            "category": category,
            "subcategory": tags[0] if tags else "General Support",
            "priority": priority,
            "intent": intent,
            "environment": "Production",
            "status": "Resolved" if answer else "New",
            "assignment_group": queue,
            "data_access_level": "internal",
            "resolution_text": answer or "Issue analyzed; standard troubleshooting procedure applied.",
            "created_at": "2026-01-15T08:00:00Z",
            "resolved_at": "2026-01-15T12:00:00Z" if answer else None
        }
        records.append(record)

        if limit and len(records) >= limit:
            break

    print(f"[+] Ingested and mapped {len(records)} tickets (Language: {lang}).")

    # Output paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

    dataset_dir = os.path.join(workspace_root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    cleaned_path = os.path.join(dataset_dir, "historical_tickets_cleaned.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[+] Saved cleaned dataset to: {cleaned_path}")

    # Prepare train/val/test splits
    splits_dir = os.path.join(dataset_dir, "splits")
    prepare_splits(cleaned_path, splits_dir)

    # Seed subset for MongoDB (up to 500 tickets for seed.js)
    seed_subset = records[:500]
    seed_path = os.path.join(workspace_root, "backend", "src", "database", "seed", "sample-tickets.json")
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(seed_subset, f, indent=2)
    print(f"[+] Updated backend sample-tickets.json with {len(seed_subset)} real tickets for MongoDB seeding.")

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Hugging Face customer support tickets")
    parser.add_argument("--lang", default="en", help="Language filter ('en', 'de', or 'all')")
    parser.add_argument("--limit", type=int, default=2000, help="Number of tickets to ingest (0 for all)")
    args = parser.parse_args()

    limit_val = None if args.limit == 0 else args.limit
    ingest_dataset(lang=args.lang, limit=limit_val)
