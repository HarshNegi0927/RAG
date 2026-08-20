"""
Live end-to-end test of grounded generation. Put your real GROQ_API_KEY in
.env (copy .env.example if you haven't) -- it's loaded automatically, no
export needed. This can't run inside the sandbox this project was built in
(api.groq.com is not reachable from there).

    python3 scripts/test_generation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from generate import answer_question  # noqa: E402

TEST_QUESTIONS = [
    "What environment variable controls the database migration timeout, and what's the default?",
    "How do I immediately disable a broken feature flag during an incident?",
    "What's the company's total revenue last quarter?",  # deliberately unanswerable -- watch what it does
]


def main():
    retriever, chunks = build_retriever()
    print(f"Indexed {len(chunks)} chunks. Generating with provider from config.py...\n")

    for q in TEST_QUESTIONS:
        retrieved = retriever.hybrid(q, top_k=5)
        result = answer_question(q, retrieved)

        print("=" * 90)
        print(f"Q: {q}")
        print("-" * 90)
        print(result.answer_text)
        print("-" * 90)
        print(f"Citations resolved: {[(c.marker, c.source_file) for c in result.citations]}")
        if result.invalid_citation_markers:
            print(f"!! Model cited markers that weren't offered as context: {result.invalid_citation_markers}")
        print(f"Provider: {result.provider} / {result.model}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
