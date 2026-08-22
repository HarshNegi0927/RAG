"""
Pre-generation confidence gate.

Kept deliberately separate from generate.py rather than baked into
answer_question() -- this is a composable pre-check any caller can apply
(eval harness, API layer, CLI script) before deciding whether to spend an
LLM call at all, not a hidden step inside generation.

Why raw BM25, not the hybrid RRF score: measured on eval/golden_qa.json,
the RRF-fused score does NOT separate answerable from unanswerable
questions (both distributions ranged ~0.011-0.016, fully overlapping) --
RRF encodes rank agreement between retrievers, not absolute relevance, so
a question with nothing relevant in the corpus can still produce a
decent-looking fused score just from winning the "best of a bad set" rank
race. Raw top-1 BM25 score, taken before fusion, does separate the two
groups (imperfectly) because a genuinely unrelated question tends to share
very little vocabulary with anything in the corpus. See
scripts/tune_confidence_threshold.py for the full measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import CONFIDENCE_SPARSE_THRESHOLD, HYBRID_TOP_K
from retrieval import HybridRetriever


@dataclass
class ConfidenceAssessment:
    query: str
    top_sparse_score: float
    threshold: float
    is_confident: bool


def assess_confidence(
    retriever: HybridRetriever, query: str, threshold: float = CONFIDENCE_SPARSE_THRESHOLD
) -> ConfidenceAssessment:
    top = retriever.sparse_only(query, top_k=1)
    score = top[0].score if top else 0.0
    return ConfidenceAssessment(query=query, top_sparse_score=score, threshold=threshold, is_confident=score >= threshold)


DECLINE_TEMPLATE = (
    "I don't have enough relevant information in the indexed documents to answer this confidently "
    "(top relevance score {score:.2f} was below the {threshold:.2f} confidence threshold). "
    "This was decided before calling the LLM, based on retrieval signal alone."
)


def answer_with_gate(retriever: HybridRetriever, query: str, top_k: int = HYBRID_TOP_K, generator=None):
    """Composes the confidence gate with generation: skip the LLM call
    entirely (real cost savings, not just a nicer refusal) when retrieval
    confidence is low, otherwise generate normally. Returns
    (GroundedAnswer, ConfidenceAssessment) so callers can see why."""
    from generate import GroundedAnswer, answer_question  # local import: avoids a hard dependency

    assessment = assess_confidence(retriever, query)
    if not assessment.is_confident:
        declined = GroundedAnswer(
            query=query,
            answer_text=DECLINE_TEMPLATE.format(score=assessment.top_sparse_score, threshold=assessment.threshold),
            citations=[],
            invalid_citation_markers=[],
            context_chunk_ids=[],
            provider="confidence_gate",
            model="none (LLM call skipped)",
        )
        return declined, assessment

    retrieved = retriever.hybrid(query, top_k=top_k)
    return answer_question(query, retrieved, generator=generator), assessment
