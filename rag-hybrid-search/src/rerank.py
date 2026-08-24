"""
Reranker. A real cross-encoder (e.g. sentence-transformers'
cross-encoder/ms-marco-MiniLM-L-6-v2) needs a HuggingFace Hub download,
which this build sandbox can't reach (same restriction documented in
embeddings.py). LLM-as-judge is the fallback the original project spec
explicitly names as valid, and it's a legitimate real technique -- ask
the generation model itself to score each candidate's relevance to the
query, then re-sort by that score.

What this fixes vs. doesn't: reranking improves PRECISION among chunks
that were already retrieved -- it re-orders the candidate set the fusion
step produced. It does NOT improve RECALL -- it can't promote a chunk
that never made the candidate set in the first place. The dep-01 rollback
miss documented in README.md's Finding #4 was a recall problem (the right
chunk ranked #24, nowhere near the top-20 candidate pool this reranks),
so reranking would not have fixed that specific case. It's still worth
having: hybrid retrieval's MRR (0.808, see README) is well short of 1.0,
meaning even on questions it answers correctly, the right chunk often
isn't ranked #1 -- that gap is exactly what reranking targets.
"""

from __future__ import annotations

import json
import re

from generate import Generator, get_generator
from retrieval import RetrievedChunk

RERANK_SYSTEM_PROMPT = """You score how relevant each numbered passage is to a query, on a 1-5 scale.
5 = directly and completely answers the query
3 = related to the query's topic but doesn't fully answer it
1 = not relevant to the query

Respond with ONLY a JSON object mapping each passage number (as a string) to its integer score. Example: {"1": 4, "2": 1, "3": 5}
No other text, no markdown code fences, just the raw JSON object."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_scores(raw: str, n_candidates: int) -> dict[int, int]:
    """Defensive parsing: models sometimes wrap JSON in ```json fences
    despite being told not to. Extract the first {...} block and parse
    that, rather than trusting raw.strip() to already be clean JSON."""
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in reranker response: {raw!r}")
    parsed = json.loads(match.group(0))
    scores: dict[int, int] = {}
    for key, value in parsed.items():
        try:
            idx = int(key)
            score = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n_candidates:
            scores[idx] = max(1, min(5, score))  # clamp defensively
    return scores


def rerank(
    query: str, candidates: list[RetrievedChunk], keep_top_k: int = 5, generator: Generator | None = None
) -> list[RetrievedChunk]:
    """Re-scores `candidates` (expected: hybrid top-20) jointly against the
    query in a single LLM call, returns the best keep_top_k re-sorted by
    that score. Falls back to the original fusion order for any candidate
    the model's response didn't score (defensive, not expected in normal
    operation)."""
    if not candidates:
        return []

    generator = generator or get_generator()
    blocks = "\n\n".join(f"[{i + 1}] {rc.chunk.text}" for i, rc in enumerate(candidates))
    user_prompt = f"Query: {query}\n\nPassages:\n\n{blocks}"

    raw = generator.complete(RERANK_SYSTEM_PROMPT, user_prompt)
    scores = _parse_scores(raw, len(candidates))

    # original_rank as tiebreak/fallback preserves the fusion ordering for
    # anything the reranker didn't score, instead of arbitrarily reshuffling it
    def sort_key(indexed):
        i, rc = indexed
        return (-scores.get(i + 1, 0), i)

    ranked = sorted(enumerate(candidates), key=sort_key)
    return [rc for _, rc in ranked[:keep_top_k]]
