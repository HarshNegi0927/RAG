"""
Live test of the confidence gate + generation composed together. Needs
GROQ_API_KEY in .env, same as the other test_*.py scripts -- except this
one should make FEWER LLM calls than test_generation.py for the same
question list, because the low-confidence question gets caught before
ever reaching Groq.

    python3 scripts/test_confidence_gate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from confidence import answer_with_gate  # noqa: E402

TEST_QUESTIONS = [
    "What environment variable controls the database migration timeout, and what's the default?",
    "What's the company's total revenue last quarter?",  # should be gated -- no LLM call at all
    "How long are our application logs kept?",  # the known false-alarm case -- watch this one
]


def main():
    retriever, chunks = build_retriever()

    for q in TEST_QUESTIONS:
        answer, assessment = answer_with_gate(retriever, q)
        print("=" * 90)
        print(f"Q: {q}")
        print(f"Confidence: top BM25={assessment.top_sparse_score:.2f}, threshold={assessment.threshold:.2f}, "
              f"confident={assessment.is_confident}")
        print("-" * 90)
        print(answer.answer_text)
        print(f"Provider: {answer.provider} / {answer.model}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
