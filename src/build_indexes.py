"""
build_indexes.py
================
Offline Index Construction Script for Semantic (FAISS) and Lexical (BM25) Retrieval.

Reads: dataset/historical_tickets_cleaned.json (400 tickets)
Produces:
- ml-service/models/faiss_index.bin (or .npy vector matrix)
- ml-service/models/ticket_id_map.json
- ml-service/models/bm25_index.joblib
- ml-service/models/ticket_metadata.json
"""

import os
import json
import logging
from embedding_service import EmbeddingService, format_ticket_for_retrieval
from faiss_index import FaissIndexManager
from bm25_service import BM25Manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_indexes")

def build_all_indexes(dataset_path: str, output_model_dir: str):
    logger.info(f"Loading historical tickets from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    logger.info(f"Loaded {len(tickets)} tickets.")

    # 1. Format text representations strictly describing problem symptoms
    # (strictly omitting resolution_text and category)
    ticket_ids = []
    retrieval_texts = []
    metadata_catalog = {}

    for t in tickets:
        tid = t["ticket_id"]
        ticket_ids.append(tid)
        text = format_ticket_for_retrieval(t)
        retrieval_texts.append(text)
        metadata_catalog[tid] = {
            "ticket_id": tid,
            "title": t.get("title", ""),
            "description": t.get("description", ""),
            "affected_service": t.get("affected_service", ""),
            "category": t.get("category", ""),
            "subcategory": t.get("subcategory", ""),
            "priority": t.get("priority", ""),
            "intent": t.get("intent", ""),
            "environment": t.get("environment", ""),
            "status": t.get("status", ""),
            "assignment_group": t.get("assignment_group", ""),
            "data_access_level": t.get("data_access_level", "internal"),
            "resolution_text": t.get("resolution_text"),
            "resolved_at": t.get("resolved_at"),
            "created_at": t.get("created_at")
        }

    os.makedirs(output_model_dir, exist_ok=True)

    # 2. Build and Persist FAISS Semantic Index
    logger.info("Generating semantic embeddings for historical tickets...")
    emb_service = EmbeddingService()
    embeddings = emb_service.encode(retrieval_texts)

    faiss_mgr = FaissIndexManager(dim=emb_service.dim)
    faiss_mgr.build_index(embeddings, ticket_ids)

    faiss_index_path = os.path.join(output_model_dir, "faiss_index.bin")
    id_map_path = os.path.join(output_model_dir, "ticket_id_map.json")
    faiss_mgr.save(faiss_index_path, id_map_path)

    # 3. Build and Persist BM25 Keyword Index
    logger.info("Building BM25 keyword index...")
    bm25_mgr = BM25Manager(k1=1.5, b=0.75)
    bm25_mgr.build_index(retrieval_texts, ticket_ids)

    bm25_path = os.path.join(output_model_dir, "bm25_index.joblib")
    bm25_mgr.save(bm25_path)

    # 4. Persist Ticket Metadata Catalog
    meta_path = os.path.join(output_model_dir, "ticket_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_catalog, f, indent=2)
    logger.info(f"Saved ticket metadata catalog to {meta_path}")

    logger.info("======================================================")
    logger.info("   All Retrieval Indexes Built and Persisted Cleanly  ")
    logger.info("======================================================")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, "..", "..", "dataset", "historical_tickets_cleaned.json")
    model_dir = os.path.join(base_dir, "..", "models")
    build_all_indexes(dataset_file, model_dir)
