"""
Full pipeline eval: confidence gate -> generation -> citation verification
-> answer correctness, combined into ONE report. The other eval scripts
each measure one layer in isolation (retrieval quality, chunking
strategy, RRF weight); this one measures whether the whole stack, wired
together, actually produces correct, well-grounded answers.

Runs on a stratified sample (2 easy + 2 medium + all 5 hard/multi-hop +
all 5 unanswerable = 14 questions), not the full 49 -- this makes real
LLM calls (generation, citation verification, correctness judging), so
the full golden set would be a lot of unnecessary API spend for a check
that's about pipeline correctness, not statistical precision. Needs
GROQ_API_KEY in .env.

    python3 scripts/run_full_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from confidence import answer_with_gate  # noqa: E402
from verify import verify_citations, citation_accuracy  # noqa: E402
from generate import get_generator  # noqa: E402
from config import EVAL_DIR  # noqa: E402

CORRECTNESS_SYSTEM_PROMPT = """You check whether a generated answer correctly conveys a specific expected fact. You will be given the expected fact and the generated answer.

Respond in exactly this format:
Line 1: one word -- CORRECT, PARTIAL, or WRONG
Line 2: one sentence explaining why

CORRECT: the answer states the expected fact accurately (paraphrasing is fine, the specific substance must match).
PARTIAL: the answer is in the right area but is missing, vague about, or slightly off on the specific fact.
WRONG: the answer contradicts the expected fact, or the expected fact isn't present in the answer at all."""


def sample_golden_set(golden: list[dict]) -> list[dict]:
    by_difficulty: dict[str, list[dict]] = {}
    for item in golden:
        by_difficulty.setdefault(item["difficulty"], []).append(item)

    sample = []
    sample += by_difficulty.get("easy", [])[:2]
    sample += by_difficulty.get("medium", [])[:2]
    sample += by_difficulty.get("hard", [])  # includes multi-hop -- only 5, take all
    sample += by_difficulty.get("unanswerable", [])  # only 5, take all
    return sample


def judge_correctness(generator, expected_note: str, generated_answer: str) -> tuple[str, str]:
    user_prompt = f"Expected fact (from notes, not shown to the generator): {expected_note}\n\nGenerated answer: {generated_answer}"
    raw = generator.complete(CORRECTNESS_SYSTEM_PROMPT, user_prompt)
    lines = raw.strip().split("\n", 1)
    verdict = lines[0].strip().upper()
    reasoning = lines[1].strip() if len(lines) > 1 else ""
    return (verdict if verdict in ("CORRECT", "PARTIAL", "WRONG") else "ERROR"), reasoning


def main():
    retriever, chunks = build_retriever()
    generator = get_generator()

    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        golden = json.load(f)
    sample = sample_golden_set(golden)
    print(f"Running full pipeline eval on {len(sample)} stratified questions "
          f"({sum(1 for s in sample if s['difficulty']=='easy')} easy, "
          f"{sum(1 for s in sample if s['difficulty']=='medium')} medium, "
          f"{sum(1 for s in sample if s['difficulty']=='hard')} hard/multi-hop, "
          f"{sum(1 for s in sample if s['difficulty']=='unanswerable')} unanswerable)\n")

    gate_correct = 0
    all_citation_results = []
    correctness_counts = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "ERROR": 0, "N/A (gated)": 0}

    for item in sample:
        q, is_answerable = item["question"], bool(item["expected_sources"])
        answer, assessment, retrieved = answer_with_gate(retriever, q)
        was_gated = answer.provider == "confidence_gate"

        gate_was_right = (was_gated and not is_answerable) or (not was_gated and is_answerable)
        gate_correct += int(gate_was_right)

        print("=" * 90)
        print(f"[{item['id']}] ({item['difficulty']}) {q}")
        print(f"  gate: score={assessment.top_sparse_score:.2f} thresh={assessment.threshold:.2f} "
              f"-> {'GATED' if was_gated else 'passed'}  [{'correct' if gate_was_right else 'WRONG gate decision'}]")

        if was_gated:
            correctness_counts["N/A (gated)"] += 1
            print(f"  answer: {answer.answer_text}")
            print("=" * 90 + "\n")
            continue

        print(f"  answer: {answer.answer_text}")

        citations = verify_citations(answer.answer_text, retrieved)
        all_citation_results.extend(citations)
        for c in citations:
            print(f"  citation [{c.verdict}]: {c.claim_text[:60]}...")

        if is_answerable:
            verdict, reasoning = judge_correctness(generator, item["notes"], answer.answer_text)
            correctness_counts[verdict] = correctness_counts.get(verdict, 0) + 1
            print(f"  correctness: {verdict} -- {reasoning}")

        print("=" * 90 + "\n")

    print("\n" + "#" * 90)
    print("SUMMARY")
    print("#" * 90)
    print(f"Confidence gate decisions correct: {gate_correct}/{len(sample)}")
    print(f"Citation verification: {citation_accuracy(all_citation_results)}")
    print(f"Answer correctness vs golden notes: {correctness_counts}")


if __name__ == "__main__":
    main()
