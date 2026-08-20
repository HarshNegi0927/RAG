"""
Citation verification: parse_citations() (in generate.py) only checks that
a citation number resolves to a real context block. It says nothing about
whether that block actually *supports* the specific claim it's attached
to. This module closes that gap with an LLM-as-judge pass, one claim at a
time.

NOT LIVE-TESTED IN THIS ENVIRONMENT for the same reason as generate.py --
neither api.groq.com nor api.x.ai is reachable from this build sandbox.
Claim-splitting (the part that doesn't need network access) IS tested
below, against real answer text from a live run on the user's machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from generate import Generator, get_generator, build_context_blocks, normalize_citation_brackets
from retrieval import RetrievedChunk

# Split on sentence-ending punctuation followed by whitespace + a capital
# letter/quote/bracket. Simple on purpose -- it'll mishandle some
# abbreviations and decimals, which is an acceptable V1 tradeoff since a
# missed split just means two claims get verified together as one, not a
# silent wrong answer.
# Split on sentence-ending punctuation followed by whitespace + the start
# of a new sentence -- but NOT when that "start" is actually a citation
# marker like [1] that got separated from the sentence it belongs to by a
# space (observed real pattern: "...incident. [1]" as well as
# "...incident[1]."). \[(?!\d) means "a bracket NOT immediately followed
# by a digit" -- that's the signal this is genuinely a new sentence
# starting with a bracket, not a trailing citation.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"]|\[(?!\d))')
_TRAILING_CITATIONS_RE = re.compile(r"((?:\[\d+\])+)[.!?]?\s*$")
_MARKER_RE = re.compile(r"\[(\d+)\]")
_MIN_CLAIM_CHARS = 10  # guard against a near-empty "claim" slipping through and getting a meaningless judge verdict


@dataclass
class Claim:
    text: str  # the sentence, with trailing citation markers stripped
    markers: list[str]
    cited_sources: list[str]
    cited_chunk_texts: list[str]


@dataclass
class VerificationResult:
    claim_text: str
    markers: list[str]
    sources: list[str]
    verdict: str  # "supported" | "partial" | "unsupported" | "error"
    reasoning: str


def split_into_claims(answer_text: str, index_map: dict[int, RetrievedChunk]) -> list[Claim]:
    """Every sentence that ends in one or more [N] markers becomes a Claim.
    Sentences with no citation are skipped here -- that's a separate,
    cheaper check (does every factual sentence have a citation at all),
    not what this function verifies."""
    answer_text = normalize_citation_brackets(answer_text)
    sentences = _SENTENCE_SPLIT_RE.split(answer_text.strip())

    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        m = _TRAILING_CITATIONS_RE.search(sent)
        if not m:
            continue
        markers = [f"[{n}]" for n in _MARKER_RE.findall(m.group(1))]
        clean_text = sent[: m.start()].strip()
        if len(clean_text) < _MIN_CLAIM_CHARS:
            continue  # nothing meaningful to verify -- e.g. citation-only fragment

        sources, chunk_texts = [], []
        for marker in markers:
            n = int(marker.strip("[]"))
            rc = index_map.get(n)
            if rc is not None:
                sources.append(rc.chunk.source_file)
                chunk_texts.append(rc.chunk.text)
        if chunk_texts:
            claims.append(Claim(clean_text, markers, sources, chunk_texts))
    return claims


VERIFICATION_SYSTEM_PROMPT = """You check whether a claim is actually supported by the source passage(s) that were cited for it.

Respond in exactly this format:
Line 1: one word -- SUPPORTED, PARTIAL, or UNSUPPORTED
Line 2: one sentence explaining why

SUPPORTED: the source passage(s) directly state or clearly imply the claim, including matching any specific numbers, names, or conditions in it.
PARTIAL: the source is on-topic and related, but doesn't fully establish the claim exactly as stated -- e.g. right subject but a different number, or the claim adds a detail the source doesn't contain.
UNSUPPORTED: the source passage(s) don't address this claim, or contradict it.

If the claim asserts that the sources do NOT contain some piece of information, and that's true of the passages shown, that counts as SUPPORTED -- absence claims are verifiable claims too."""


def verify_citations(
    answer_text: str, retrieved_chunks: list[RetrievedChunk], generator: Generator | None = None
) -> list[VerificationResult]:
    generator = generator or get_generator()
    _, index_map = build_context_blocks(retrieved_chunks)
    claims = split_into_claims(answer_text, index_map)

    results = []
    for claim in claims:
        sources_block = "\n\n".join(
            f"Source ({src}):\n{text}" for src, text in zip(claim.cited_sources, claim.cited_chunk_texts)
        )
        user_prompt = f"Claim: {claim.text}\n\nCited source passage(s):\n\n{sources_block}"
        raw = generator.complete(VERIFICATION_SYSTEM_PROMPT, user_prompt)

        lines = raw.strip().split("\n", 1)
        verdict_word = lines[0].strip().upper()
        reasoning = lines[1].strip() if len(lines) > 1 else ""
        verdict = verdict_word.lower() if verdict_word in ("SUPPORTED", "PARTIAL", "UNSUPPORTED") else "error"

        results.append(VerificationResult(claim.text, claim.markers, claim.cited_sources, verdict, reasoning))
    return results


def citation_accuracy(results: list[VerificationResult]) -> dict:
    if not results:
        return {"supported": 0, "partial": 0, "unsupported": 0, "error": 0, "total": 0, "accuracy": None}
    counts = {"supported": 0, "partial": 0, "unsupported": 0, "error": 0}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    total = len(results)
    # partial counts as half credit -- it's not wrong, but it's not clean either
    accuracy = (counts["supported"] + 0.5 * counts["partial"]) / total
    return {**counts, "total": total, "accuracy": round(accuracy, 3)}
