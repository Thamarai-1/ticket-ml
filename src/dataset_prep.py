"""
dataset_prep.py
===============
Stratified Dataset Preparation and Leakage Prevention.

Requirements:
1. Fixed random seed (42).
2. Stratified class distribution across splits.
3. Completely separate held-out test set (15%) and validation set (15%).
4. Input features: title, description, affected_service.
5. Target: category.
6. NO post-resolution fields (resolution_text, resolved_at, status) in features.
"""

import json
import os
import random
from collections import defaultdict
from preprocessing import combine_ticket_text

RANDOM_SEED = 42

def prepare_splits(dataset_path: str, output_dir: str):
    random.seed(RANDOM_SEED)

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    cat_map = defaultdict(list)
    for r in records:
        text_feature = combine_ticket_text(
            r.get("title", ""),
            r.get("description", ""),
            r.get("affected_service", "")
        )
        sample = {
            "ticket_id": r["ticket_id"],
            "title": r["title"],
            "description": r["description"],
            "affected_service": r.get("affected_service", ""),
            "category": r["category"],
            "subcategory": r.get("subcategory"),
            "priority": r.get("priority"),
            "intent": r.get("intent"),
            "text": text_feature
        }
        cat_map[r["category"]].append(sample)

    train_data = []
    val_data = []
    test_data = []

    for cat, items in cat_map.items():
        random.shuffle(items)
        n_items = len(items)
        n_train = int(round(n_items * 0.70))
        n_val = int(round(n_items * 0.15))
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"[Dataset Preparation Complete] Seed: {RANDOM_SEED}")
    print(f"  - Total: {len(records)}")
    print(f"  - Train: {len(train_data)} ({len(train_data)/len(records)*100:.1f}%)")
    print(f"  - Val:   {len(val_data)} ({len(val_data)/len(records)*100:.1f}%)")
    print(f"  - Test:  {len(test_data)} ({len(test_data)/len(records)*100:.1f}%)")

    return train_data, val_data, test_data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, "..", "..", "dataset", "historical_tickets_cleaned.json")
    output_dir = os.path.join(base_dir, "..", "..", "dataset", "splits")
    prepare_splits(dataset_file, output_dir)
