"""
bm25_service.py
===============
BM25 Keyword Information Retrieval Module.

Features:
- Technical tokenization preserving hexadecimal error codes, IP addresses, and file paths
- BM25Okapi scoring (k1=1.5, b=0.75)
- Native rank_bm25 integration with pure-python BM25Okapi fallback
- Serialization to disk via joblib
"""

import os
import re
import math
import logging
import joblib

logger = logging.getLogger("ml-service.bm25")

def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenizes text while strictly preserving:
    - Hexadecimal error codes (0x80070005)
    - Dotted IP addresses and ports (10.240.0.1, 127.0.0.1:8080)
    - Windows Event IDs and stop codes
    - Dotted system files (outlmapi.dll, nvlddmkm.sys)
    """
    if not text:
        return []
    text_lower = text.lower()
    # Match hex codes, dotted IPs/filenames, or standard words
    tokens = re.findall(r"0x[0-9a-fA-F]+|\b\w+(?:[\.\-]\w+)*\b", text_lower)
    return tokens

class SimpleBM25Okapi:
    """Pure-Python BM25Okapi fallback implementation with zero external dependencies."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs = []
        self.idf = {}
        self._init_idf(corpus)

    def _init_idf(self, corpus):
        df = {}
        for doc in corpus:
            doc_set = set(doc)
            for token in doc_set:
                df[token] = df.get(token, 0) + 1
        for token, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        query_set = set(query)
        for token in query_set:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, doc_freqs in enumerate(self.doc_freqs):
                freq = doc_freqs.get(token, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_doc_len))
                scores[doc_idx] += idf_val * (num / denom)
        return scores

class BM25Manager:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.id_map = []
        self.tokenized_corpus = []

    def build_index(self, texts: list[str], ticket_ids: list[str]):
        assert len(texts) == len(ticket_ids)
        self.id_map = list(ticket_ids)
        self.tokenized_corpus = [tokenize_for_bm25(t) for t in texts]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.info(f"Built native rank_bm25 BM25Okapi with {len(self.tokenized_corpus)} documents.")
        except ImportError:
            self.bm25 = SimpleBM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            # Pre-compute doc freqs for fallback
            self.bm25.doc_freqs = []
            for doc in self.tokenized_corpus:
                counts = {}
                for t in doc:
                    counts[t] = counts.get(t, 0) + 1
                self.bm25.doc_freqs.append(counts)
            logger.info(f"Built pure-python BM25Okapi fallback with {len(self.tokenized_corpus)} documents.")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float, int]]:
        """
        Calculates BM25 lexical scores for query.
        Returns list of (ticket_id, bm25_score, rank).
        """
        if not self.bm25 or len(self.id_map) == 0:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            score = float(scores[idx])
            if score > 0.0:  # Only return tickets with at least 1 keyword match
                results.append((self.id_map[idx], round(score, 4), rank))

        return results

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "id_map": self.id_map,
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved BM25 state to {filepath}")

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            logger.warning(f"BM25 artifact path does not exist: {filepath}")
            return False

        state = joblib.load(filepath)
        self.id_map = state["id_map"]
        self.tokenized_corpus = state["tokenized_corpus"]
        self.k1 = state.get("k1", 1.5)
        self.b = state.get("b", 0.75)

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        except ImportError:
            self.bm25 = SimpleBM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            self.bm25.doc_freqs = []
            for doc in self.tokenized_corpus:
                counts = {}
                for t in doc:
                    counts[t] = counts.get(t, 0) + 1
                self.bm25.doc_freqs.append(counts)

        logger.info(f"Loaded BM25 index with {len(self.id_map)} tickets.")
        return True
