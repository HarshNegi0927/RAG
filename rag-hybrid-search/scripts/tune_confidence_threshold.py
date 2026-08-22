"""
Measures the confidence gate against the full golden set: how many
unanswerable questions get correctly caught, and how many genuinely
answerable questions get wrongly blocked, at the configured threshold.
Pure retrieval -- no API key needed, runs fully offline.

Run: python3 scripts/tune_confidence_threshold.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from confidence import assess_confidence  # noqa: E402
from config import EVAL_DIR, CONFIDENCE_SPARSE_THRESHOLD  # noqa: E402


def main():
    retriever, _ = build_retriever()
    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        golden = json.load(f)

    print(f"Threshold in use (config.CONFIDENCE_SPARSE_THRESHOLD): {CONFIDENCE_SPARSE_THRESHOLD}\n")

    correctly_blocked, wrongly_allowed, correctly_allowed, wrongly_blocked = [], [], [], []
    for item in golden:
        assessment = assess_confidence(retriever, item["question"])
        is_answerable = bool(item["expected_sources"])
        row = (item["id"], item["question"][:55], round(assessment.top_sparse_score, 2))
        if is_answerable and assessment.is_confident:
            correctly_allowed.append(row)
        elif is_answerable and not assessment.is_confident:
            wrongly_blocked.append(row)
        elif not is_answerable and not assessment.is_confident:
            correctly_blocked.append(row)
        else:
            wrongly_allowed.append(row)

    print(f"Correctly blocked (unanswerable, caught):   {len(correctly_blocked)}/5")
    for qid, q, s in correctly_blocked:
        print(f"    [{s:>5}] {qid:<10} {q}")

    print(f"\nWrongly allowed (unanswerable, missed):      {len(wrongly_allowed)}/5")
    for qid, q, s in wrongly_allowed:
        print(f"    [{s:>5}] {qid:<10} {q}")

    print(f"\nWrongly blocked (answerable, false alarm):   {len(wrongly_blocked)}/44")
    for qid, q, s in wrongly_blocked:
        print(f"    [{s:>5}] {qid:<10} {q}")

    print(f"\nCorrectly allowed (answerable, passed):      {len(correctly_allowed)}/44")


if __name__ == "__main__":
    main()
