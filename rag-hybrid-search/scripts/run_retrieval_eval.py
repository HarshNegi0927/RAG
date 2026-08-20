"""
Retrieval evaluation harness.

Runs every question in eval/golden_qa.json through dense-only, sparse-only,
and hybrid retrieval, and reports:
  - Hit@k   : was at least one chunk from EVERY expected source doc present
              in the top-k results? (for multi-hop questions this requires
              ALL expected sources to show up, which is a stricter and more
              honest bar than "any one of them")
  - MRR     : mean reciprocal rank, using the rank of the LAST (worst)
              expected source to appear -- you can't actually answer a
              multi-hop question until you've found all the pieces, so
              scoring on the easiest piece would be misleading.

Unanswerable questions (expected_sources: []) are reported separately --
there's no "correct doc" to score retrieval against, but tracking what
IS returned for them matters later, once generation is wired up and needs
to recognize "the docs don't actually cover this" instead of confidently
grounding an answer in the closest-but-wrong chunk.

Run: python3 scripts/run_retrieval_eval.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from config import EVAL_DIR  # noqa: E402

TOP_K = 5


def load_golden_set() -> list[dict]:
    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        return json.load(f)


def score_one(results, expected_sources: list[str]) -> tuple[bool, float]:
    """Returns (hit, reciprocal_rank) for a single retrieval result list
    against one QA pair's expected_sources."""
    if not expected_sources:
        return None, None  # unanswerable case, scored separately

    rank_of_source: dict[str, int | None] = {src: None for src in expected_sources}
    for r in results:
        if r.chunk.source_file in rank_of_source and rank_of_source[r.chunk.source_file] is None:
            rank_of_source[r.chunk.source_file] = r.rank

    ranks = list(rank_of_source.values())
    hit = all(r is not None for r in ranks)
    if not hit:
        return False, 0.0
    worst_rank = max(ranks)  # need every expected source found; score by the hardest one
    return True, 1.0 / worst_rank


def run_eval():
    golden = load_golden_set()
    retriever, chunks = build_retriever()
    print(f"Loaded {len(golden)} golden QA pairs. Corpus: {len(chunks)} chunks.\n")

    methods = ["dense", "sparse", "hybrid"]
    # results[method][difficulty] = list of (hit, rr)
    by_difficulty = {m: defaultdict(list) for m in methods}
    by_kind = {m: defaultdict(list) for m in methods}
    overall = {m: [] for m in methods}
    unanswerable_leakage = {m: [] for m in methods}  # what DOES come back for unanswerable Qs

    for item in golden:
        q = item["question"]
        expected = item["expected_sources"]

        run = {
            "dense": retriever.dense_only(q, top_k=TOP_K),
            "sparse": retriever.sparse_only(q, top_k=TOP_K),
            "hybrid": retriever.hybrid(q, top_k=TOP_K),
        }

        for m in methods:
            if not expected:
                top = run[m][0] if run[m] else None
                unanswerable_leakage[m].append(top.chunk.source_file if top else None)
                continue
            hit, rr = score_one(run[m], expected)
            by_difficulty[m][item["difficulty"]].append((hit, rr))
            by_kind[m][item["kind"]].append((hit, rr))
            overall[m].append((hit, rr))

    def pct(vals, idx):
        xs = [v[idx] for v in vals]
        return 100.0 * sum(1 for x in xs if x) / len(xs) if idx == 0 else sum(xs) / len(xs)

    print("=" * 78)
    print(f"OVERALL (n={len(overall['dense'])} answerable questions, top-{TOP_K})")
    print("=" * 78)
    print(f"{'Method':<10} {'Hit@' + str(TOP_K):<12} {'MRR':<10}")
    for m in methods:
        hit_rate = pct(overall[m], 0)
        mrr = pct(overall[m], 1)
        print(f"{m:<10} {hit_rate:>6.1f}%     {mrr:>6.3f}")

    print("\n" + "=" * 78)
    print(f"BY DIFFICULTY (Hit@{TOP_K})")
    print("=" * 78)
    difficulties = ["easy", "medium", "hard"]
    print(f"{'Method':<10}" + "".join(f"{d:<12}" for d in difficulties))
    for m in methods:
        row = f"{m:<10}"
        for d in difficulties:
            vals = by_difficulty[m].get(d, [])
            row += f"{pct(vals, 0):>6.1f}%     " if vals else f"{'n/a':<12}"
        print(row)

    print("\n" + "=" * 78)
    print(f"BY QUERY KIND (Hit@{TOP_K})")
    print("=" * 78)
    kinds = ["identifier", "paraphrase", "multi-hop"]
    print(f"{'Method':<10}" + "".join(f"{k:<14}" for k in kinds))
    for m in methods:
        row = f"{m:<10}"
        for k in kinds:
            vals = by_kind[m].get(k, [])
            row += f"{pct(vals, 0):>6.1f}%       " if vals else f"{'n/a':<14}"
        print(row)

    print("\n" + "=" * 78)
    print(f"UNANSWERABLE QUESTIONS (n={len(unanswerable_leakage['dense'])}) -- top-1 source returned")
    print("Retrieval will always return SOMETHING here (there's no threshold gate yet).")
    print("This is expected -- it's exactly the gap the confidence/'I don't know' layer")
    print("in the generation step needs to close. Recorded here as a baseline.")
    print("=" * 78)
    for m in methods:
        print(f"{m}: {unanswerable_leakage[m]}")


if __name__ == "__main__":
    run_eval()
