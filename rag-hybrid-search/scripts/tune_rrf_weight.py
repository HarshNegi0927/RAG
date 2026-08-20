"""
Weight-tuning sweep for RRF fusion.

Equal-weighted RRF (dense_weight=sparse_weight=0.5) assumes both signals
deserve equal trust. The retrieval eval showed that's not true for THIS
corpus with the free local embedder -- sparse alone beat 50/50 hybrid.
This script sweeps dense_weight from 0.0 (sparse-only) to 1.0 (dense-only)
and reports Hit@5 / MRR at each point, so the final weight in config is a
measured choice, not a guess.

Run: python3 scripts/tune_rrf_weight.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from config import EVAL_DIR  # noqa: E402
from run_retrieval_eval import score_one, TOP_K  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow the import above


def main():
    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        golden = [g for g in json.load(f) if g["expected_sources"]]  # skip unanswerable

    retriever, _ = build_retriever()

    weights_to_try = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    print(f"Sweeping dense_weight over {weights_to_try} (sparse_weight = 1 - dense_weight)")
    print(f"n={len(golden)} answerable questions, top-{TOP_K}\n")
    print(f"{'dense_weight':<14}{'sparse_weight':<15}{'Hit@' + str(TOP_K):<10}{'MRR':<8}")

    best = None
    for dw in weights_to_try:
        sw = round(1.0 - dw, 2)
        hits, rrs = [], []
        for item in golden:
            results = retriever.hybrid(item["question"], top_k=TOP_K, dense_weight=dw, sparse_weight=sw)
            hit, rr = score_one(results, item["expected_sources"])
            hits.append(hit)
            rrs.append(rr)
        hit_rate = 100.0 * sum(hits) / len(hits)
        mrr = sum(rrs) / len(rrs)
        marker = ""
        if best is None or (hit_rate, mrr) > (best[1], best[2]):
            best = (dw, hit_rate, mrr)
        print(f"{dw:<14}{sw:<15}{hit_rate:>5.1f}%    {mrr:.3f}")

    print(f"\nBest on this eval set: dense_weight={best[0]} "
          f"(Hit@{TOP_K}={best[1]:.1f}%, MRR={best[2]:.3f})")
    print("\nNote: at dense_weight=0.0 this is literally sparse-only, which is the honest")
    print("baseline to beat. The real claim to make in a writeup is whichever weight here")
    print("beats that baseline, by how much, and only for the corpus/embedder actually tested.")


if __name__ == "__main__":
    main()
