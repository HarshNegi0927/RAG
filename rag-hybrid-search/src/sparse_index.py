"""
BM25 sparse index. This is the layer that catches exact identifiers --
env var names, function names, error codes -- that word/semantic vectors
smear across their neighborhood instead of matching precisely. See
scripts/compare_retrieval.py for a concrete example (`DB_MIGRATION_TIMEOUT_SEC`).
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokenizer. Keeps underscores intact so
    identifiers like DB_MIGRATION_TIMEOUT_SEC stay as one meaningful token
    family instead of being shredded into db/migration/timeout/sec."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseIndex:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids = chunk_ids
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted descending, length <= top_k."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]
