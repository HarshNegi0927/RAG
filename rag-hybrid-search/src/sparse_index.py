"""
BM25 sparse index. This is the layer that catches exact identifiers --
env var names, function names, error codes -- that word/semantic vectors
smear across their neighborhood instead of matching precisely. See
scripts/compare_retrieval.py for a concrete example (`DB_MIGRATION_TIMEOUT_SEC`).

Tokens are stemmed (Porter stemmer) before indexing and querying, so
morphological variants match each other -- "rotate"/"rotation"/"rotating"
all reduce to the same stem. This does NOT fix every mismatch: a real
failure found via scripts/run_full_eval.py was the query "roll back"
(two words) against the doc's "rollback" (one word) -- these tokenize to
completely different tokens (['roll','back'] vs ['rollback']) and no
stemmer merges separate words into a compound one. That specific class of
miss needs either phrase/bigram indexing or subword-tokenized embeddings
(e.g. OpenAI's), not stemming -- noted here so it isn't silently assumed
fixed.
"""

from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Lowercase, stem, word/identifier tokenizer. Keeps underscores intact
    so identifiers like DB_MIGRATION_TIMEOUT_SEC stay as one meaningful
    token instead of being shredded into db/migration/timeout/sec (the
    stemmer is a no-op on these since they aren't recognized English word
    forms, which is what we want -- identifiers should match exactly)."""
    return [_stemmer.stem(t.lower()) for t in _TOKEN_RE.findall(text)]


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
    