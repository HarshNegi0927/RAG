"""
Generation: turn retrieved chunks + a question into a grounded, cited
answer.

Provider is pluggable (same pattern as embeddings.py): `Generator.complete()`
is the one method every backend implements, and `answer_question()` is the
shared logic (prompt construction, citation parsing) that doesn't change
between providers.

NOT LIVE-TESTED IN THIS ENVIRONMENT: this build sandbox's network egress
is allow-listed to specific domains, and api.x.ai is not on that list
(confirmed: the proxy returns `x-deny-reason: host_not_allowed`). The code
below is written correctly against xAI's documented OpenAI-compatible
endpoint (https://docs.x.ai) but the actual API round-trip has not been
exercised here. Run scripts/test_generation.py yourself with XAI_API_KEY
set to confirm it end to end.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import (
    GENERATION_PROVIDER,
    GROQ_MODEL,
    GROQ_BASE_URL,
    GROK_MODEL,
    ANTHROPIC_MODEL,
    OPENAI_GENERATION_MODEL,
    XAI_BASE_URL,
)
from retrieval import RetrievedChunk


# --- Provider backends -------------------------------------------------

class Generator(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn completion. Returns the raw text response."""
        ...


class GroqGenerator(Generator):
    """GroqCloud -- fast inference hosting for open models (Llama, gpt-oss,
    Qwen, etc.) on custom LPU hardware. This is the default provider.

    Not the same product as xAI's Grok (see GrokGenerator below) -- the
    names collide easily, this repo used to default to the wrong one for
    exactly that reason."""

    name = "groq"

    def __init__(self, model: str = GROQ_MODEL):
        from openai import OpenAI  # Groq ships an OpenAI-compatible API

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GENERATION_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Get one at https://console.groq.com/keys and export it."
            )
        self._client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # grounded QA wants low creativity
        )
        return resp.choices[0].message.content


class GrokGenerator(Generator):
    """xAI Grok. Different company from Groq above -- kept available since
    it's already built and tested the same way, but not the default."""

    name = "grok"

    def __init__(self, model: str = GROK_MODEL):
        from openai import OpenAI  # xAI also ships an OpenAI-compatible API

        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise RuntimeError(
                "GENERATION_PROVIDER=grok but XAI_API_KEY is not set. "
                "Get one at https://console.x.ai and export it."
            )
        self._client = OpenAI(api_key=key, base_url=XAI_BASE_URL)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content


class AnthropicGenerator(Generator):
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL):
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("GENERATION_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIGenerator(Generator):
    name = "openai"

    def __init__(self, model: str = OPENAI_GENERATION_MODEL):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("GENERATION_PROVIDER=openai but OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=key)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content


_REGISTRY = {
    "groq": GroqGenerator,
    "grok": GrokGenerator,
    "anthropic": AnthropicGenerator,
    "openai": OpenAIGenerator,
}


def get_generator(provider: str | None = None) -> Generator:
    provider = provider or GENERATION_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown generation provider '{provider}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[provider]()


# --- Prompting + citation parsing (provider-agnostic) -------------------

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context blocks provided below. Follow these rules exactly:

1. Every factual claim you make must be followed by the bracketed number(s) of the context block(s) it came from, like [1] or [2][3]. Never write a claim without a citation.
2. Citation brackets must be the plain ASCII characters [ and ] (U+005B / U+005D) -- for example [1]. Do not use full-width brackets (\u3010\u3011), fullwidth square brackets (\uFF3B\uFF3D), or any other bracket-like character.
3. If two blocks are needed to support one claim, cite both: [1][3], not [1,3].
4. If the context blocks do not contain enough information to answer the question, say so explicitly -- name what's missing -- rather than filling the gap from outside knowledge. Do not guess.
5. Do not use any knowledge beyond what's in the context blocks, even if you know the answer from elsewhere.
6. Be concise. Answer the question directly first, then add supporting detail if useful."""


# Some models (observed: Groq's gpt-oss-120b) cite using full-width or other
# bracket-look-alike characters even when the prompt explicitly asks for
# plain ASCII "[1]" -- instruction-following on this specific detail isn't
# 100% reliable. Normalize known variants to ASCII brackets before parsing
# OR displaying the answer, rather than trusting the model's literal output.
_BRACKET_VARIANTS = {
    "\u3010": "[", "\u3011": "]",  # 【 】 full-width lenticular brackets
    "\uff3b": "[", "\uff3d": "]",  # ［ ］ full-width square brackets
    "\u27e6": "[", "\u27e7": "]",  # ⟦ ⟧ mathematical white square brackets
}


def normalize_citation_brackets(text: str) -> str:
    for variant, ascii_char in _BRACKET_VARIANTS.items():
        text = text.replace(variant, ascii_char)
    return text


@dataclass
class Citation:
    marker: str  # e.g. "[1]"
    chunk_id: str
    source_file: str
    section_heading: str | None


@dataclass
class GroundedAnswer:
    query: str
    answer_text: str
    citations: list[Citation]
    invalid_citation_markers: list[str]  # model cited a number that wasn't offered as context
    context_chunk_ids: list[str]
    provider: str
    model: str


def build_context_blocks(retrieved_chunks: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
    """Returns (formatted_context_string, {block_number: RetrievedChunk})."""
    lines = []
    index_map: dict[int, RetrievedChunk] = {}
    for i, rc in enumerate(retrieved_chunks, start=1):
        heading = rc.chunk.section_heading or "(no heading)"
        lines.append(f"[{i}] (source: {rc.chunk.source_file}, section: {heading})\n{rc.chunk.text}")
        index_map[i] = rc
    return "\n\n".join(lines), index_map


_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer_text: str, index_map: dict[int, RetrievedChunk]) -> tuple[list[Citation], list[str]]:
    found_numbers = sorted({int(n) for n in _CITATION_RE.findall(answer_text)})
    citations, invalid = [], []
    for n in found_numbers:
        rc = index_map.get(n)
        if rc is None:
            invalid.append(f"[{n}]")
            continue
        citations.append(
            Citation(
                marker=f"[{n}]",
                chunk_id=rc.chunk.chunk_id,
                source_file=rc.chunk.source_file,
                section_heading=rc.chunk.section_heading,
            )
        )
    return citations, invalid


def answer_question(
    query: str, retrieved_chunks: list[RetrievedChunk], generator: Generator | None = None
) -> GroundedAnswer:
    generator = generator or get_generator()
    context_str, index_map = build_context_blocks(retrieved_chunks)
    user_prompt = f"Context blocks:\n\n{context_str}\n\nQuestion: {query}"

    raw_answer = generator.complete(SYSTEM_PROMPT, user_prompt)
    normalized_answer = normalize_citation_brackets(raw_answer)
    citations, invalid = parse_citations(normalized_answer, index_map)

    return GroundedAnswer(
        query=query,
        answer_text=normalized_answer,
        citations=citations,
        invalid_citation_markers=invalid,
        context_chunk_ids=[rc.chunk.chunk_id for rc in retrieved_chunks],
        provider=generator.name,
        model=generator.model,
    )
