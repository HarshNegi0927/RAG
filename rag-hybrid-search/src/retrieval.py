"""
Retrieval layer: dense index, RRF fusion with the sparse (BM25) index, and
a HybridRetriever that exposes dense-only / sparse-only / hybrid search
through the same interface so they're directly comparable.

Vector store note: this uses a plain numpy brute-force cosine search,
which is the right choice at this corpus size (tens of chunks -- an exact
search over a few hundred floats is microseconds, no approximate index
needed). DenseIndex is written behind a small interface on purpose so
swapping in ChromaDB or Qdrant for a larger, persistent corpus is a
one-file change, not a rewrite of retrieval.py or anything upstream of it.

RRF (Reciprocal Rank Fusion) note: it fuses by *rank position*, not raw
score, which sidesteps the problem that BM25 scores and cosine
similarities live on completely different, incomparable scales. A chunk
ranked #1 by BM25 and #3 by dense search gets
1/(k+1) + 1/(k+3); a chunk that only one method found still gets credit
from that one ranking. k=60 is the constant from the original RRF paper
and is not particularly sensitive to tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DENSE_TOP_K, SPARSE_TOP_K, RRF_K, HYBRID_TOP_K, DENSE_WEIGHT, SPARSE_WEIGHT
from ingest import Chunk
from sparse_index import SparseIndex


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    method: str  # "dense" | "sparse" | "hybrid"
    # For hybrid results, keep the underlying signals visible -- this is
    # exactly what a "why was this retrieved" explanation in the eventual
    # dashboard would show.
    dense_rank: int | None = None
    sparse_rank: int | None = None


class DenseIndex:
    def __init__(self, chunk_ids: list[str], vectors: np.ndarray):
        self.chunk_ids = chunk_ids
        self.vectors = vectors  # assumed L2-normalized rows

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        qv = query_vector / (np.linalg.norm(query_vector) or 1.0)
        sims = self.vectors @ qv  # cosine similarity since both sides are unit-norm
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunk_ids[i], float(sims[i])) for i in top_idx]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """ranked_lists: list of [(chunk_id, score), ...] already sorted best-first.
    weights: per-list multiplier (default: equal weight = standard RRF).
    Returns [(chunk_id, fused_score), ...] sorted best-first.

    Equal-weighted RRF assumes both signals are equally trustworthy. They
    aren't always -- see scripts/tune_rrf_weight.py, where a weak dense
    signal on this corpus turned out to actively drag down a strong BM25
    ranking under equal weighting. Weighting lets you say "trust sparse
    more" without abandoning the fused signal entirely."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists in length")

    fused: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense_index: DenseIndex, sparse_index: SparseIndex, embedder):
        self._by_id = {c.chunk_id: c for c in chunks}
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder

    def _wrap(self, chunk_id: str, score: float, rank: int, method: str, **kw) -> RetrievedChunk:
        return RetrievedChunk(chunk=self._by_id[chunk_id], score=score, rank=rank, method=method, **kw)

    def dense_only(self, query: str, top_k: int = DENSE_TOP_K) -> list[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        results = self.dense_index.search(qvec, top_k=top_k)
        return [self._wrap(cid, score, i + 1, "dense") for i, (cid, score) in enumerate(results)]

    def sparse_only(self, query: str, top_k: int = SPARSE_TOP_K) -> list[RetrievedChunk]:
        results = self.sparse_index.search(query, top_k=top_k)
        return [self._wrap(cid, score, i + 1, "sparse") for i, (cid, score) in enumerate(results)]

    def hybrid(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
    ) -> list[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        dense_results = self.dense_index.search(qvec, top_k=DENSE_TOP_K)
        sparse_results = self.sparse_index.search(query, top_k=SPARSE_TOP_K)

        dense_rank_of = {cid: r + 1 for r, (cid, _) in enumerate(dense_results)}
        sparse_rank_of = {cid: r + 1 for r, (cid, _) in enumerate(sparse_results)}

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results], weights=[dense_weight, sparse_weight]
        )[:top_k]
        return [
            self._wrap(
                cid, score, i + 1, "hybrid",
                dense_rank=dense_rank_of.get(cid),
                sparse_rank=sparse_rank_of.get(cid),
            )
            for i, (cid, score) in enumerate(fused)
        ]
