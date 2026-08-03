"""
resource_rag.py
Keyword retrieval (BM25 + stemming, same approach validated in the
scheme-navigator project) over the curated skill knowledge base.

This is the RAG half of the pipeline and it's deliberately the part that
does NOT touch an LLM: when report.py needs to suggest "how do I close
this gap", it retrieves a real, pre-written entry from data/skill_resources.json
rather than asking the LLM to invent a course name or resource. This is
what keeps the coaching suggestions from hallucinating -- retrieval
supplies the facts, generation only phrases them (same discipline as the
eligibility project, applied to a domain where it actually matters more,
since a fabricated course name is a much easier hallucination to produce
convincingly than a fabricated income threshold).
"""

import json
import os
import re
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skill_resources.json")


def _tokenize(text: str) -> list:
    return [_stemmer.stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


def _skill_text(skill: dict) -> str:
    return " ".join([skill["name"], skill["category"], skill["description"], skill.get("how_to_demonstrate", "")])


class ResourceIndex:
    def __init__(self, data_path: str = DATA_PATH):
        with open(data_path, encoding="utf-8") as f:
            self.skills = {s["id"]: s for s in json.load(f)["skills"]}
        self.ids = list(self.skills.keys())
        corpus = [_tokenize(_skill_text(self.skills[i])) for i in self.ids]
        self.bm25 = BM25Okapi(corpus)

    def find_resource(self, gap_description: str, top_k: int = 1) -> list:
        """Given a free-text description of a missing/partial skill (as
        identified by the LLM matcher), retrieve the closest real entries
        from the curated knowledge base."""
        scores = self.bm25.get_scores(_tokenize(gap_description))
        pairs = sorted(zip(self.ids, scores), key=lambda x: -x[1])
        return [self.skills[i] for i, s in pairs[:top_k] if s > 0]


if __name__ == "__main__":
    idx = ResourceIndex()
    for q in ["needs more production deployment experience", "no RAG or LLM project experience", "weak in SQL window functions"]:
        hits = idx.find_resource(q, top_k=2)
        print(f"\nGap: {q!r}")
        for h in hits:
            print(f"  -> {h['id']}: {h['how_to_demonstrate']}")
