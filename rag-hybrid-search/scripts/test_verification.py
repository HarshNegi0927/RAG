"""
Live end-to-end test of citation verification. Runs generation, then
verifies each cited claim against its cited source. Needs GROQ_API_KEY in
.env (same as test_generation.py). Can't run in the build sandbox -- see
that script's docstring for why.

    python3 scripts/test_verification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from generate import answer_question  # noqa: E402
from verify import verify_citations, citation_accuracy  # noqa: E402

TEST_QUESTIONS = [
    "What environment variable controls the database migration timeout, and what's the default?",
    "How do I immediately disable a broken feature flag during an incident?",
    "What's the company's total revenue last quarter?",  # the 5-citation decline case
]


def main():
    retriever, chunks = build_retriever()
    all_results = []

    for q in TEST_QUESTIONS:
        retrieved = retriever.hybrid(q, top_k=5)
        answer = answer_question(q, retrieved)
        verification = verify_citations(answer.answer_text, retrieved)
        all_results.extend(verification)

        print("=" * 90)
        print(f"Q: {q}")
        print("-" * 90)
        print(answer.answer_text)
        print("-" * 90)
        for v in verification:
            flag = {"supported": "OK", "partial": "??", "unsupported": "XX", "error": "!!"}[v.verdict]
            print(f"[{flag}] {v.verdict.upper()} -- claim: \"{v.claim_text[:70]}...\"")
            print(f"       cited: {v.sources} | reasoning: {v.reasoning}")
        print("=" * 90 + "\n")

    summary = citation_accuracy(all_results)
    print(f"OVERALL: {summary}")
    if summary["accuracy"] is not None and summary["accuracy"] < 1.0:
        print("Not every citation was fully SUPPORTED -- that's expected and useful, not a failure of")
        print("this script. Read the PARTIAL/UNSUPPORTED reasoning above to see what it caught.")


if __name__ == "__main__":
    main()
