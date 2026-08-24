"""
Measures reranking's real effect: for a sample of golden questions, ranks
the correct chunk within the hybrid top-20 BEFORE reranking, then AFTER,
and reports aggregate MRR both ways. Answers "does this actually help on
this corpus" rather than assuming it does because it's a known technique.

Runs on the "hard" + "multi-hop" questions specifically (10 total) --
these are where fusion's ranking is weakest (see README: hybrid Hit@5 by
difficulty drops to 80% on "hard" vs 100% on "easy"), so it's the most
informative subset to check for a precision fix. Needs GROQ_API_KEY.

    python3 scripts/measure_reranker_impact.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402
from rerank import rerank  # noqa: E402
from generate import get_generator  # noqa: E402
from config import EVAL_DIR, RERANK_POOL_SIZE  # noqa: E402


def rank_of_expected(chunks, expected_sources) -> int | None:
    for i, rc in enumerate(chunks, start=1):
        if rc.chunk.source_file in expected_sources:
            return i
    return None


def main():
    retriever, _ = build_retriever()
    generator = get_generator()

    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        golden = json.load(f)
    sample = [g for g in golden if g["difficulty"] == "hard"]  # includes multi-hop, per golden_qa.json's tagging

    print(f"Measuring reranker impact on {len(sample)} hard/multi-hop questions "
          f"(pool={RERANK_POOL_SIZE}, generator={generator.name}/{generator.model})\n")

    before_ranks, after_ranks = [], []
    for item in sample:
        q, expected = item["question"], item["expected_sources"]
        pool = retriever.hybrid(q, top_k=RERANK_POOL_SIZE)

        before = rank_of_expected(pool, expected)
        reranked = rerank(q, pool, keep_top_k=RERANK_POOL_SIZE, generator=generator)
        after = rank_of_expected(reranked, expected)

        before_ranks.append(1.0 / before if before else 0.0)
        after_ranks.append(1.0 / after if after else 0.0)

        print(f"[{item['id']}] {q[:55]}")
        print(f"   rank before: {before or 'not found'}   rank after: {after or 'not found'}")

    mrr_before = sum(before_ranks) / len(before_ranks)
    mrr_after = sum(after_ranks) / len(after_ranks)
    print(f"\nMRR before reranking: {mrr_before:.3f}")
    print(f"MRR after reranking:  {mrr_after:.3f}")
    print(f"{'Improved' if mrr_after > mrr_before else 'Did not improve'} on this sample.")


if __name__ == "__main__":
    main()
