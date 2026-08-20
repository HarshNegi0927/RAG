"""
Chunking strategy comparison: same corpus, same golden eval set, same
embedder, only the chunking strategy changes. Answers "does structure-aware
chunking actually retrieve better than naive fixed-size chunking, or did I
just add complexity for nothing?"

Run: python3 scripts/compare_chunking.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import build_chunks  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from sparse_index import SparseIndex  # noqa: E402
from retrieval import DenseIndex, HybridRetriever  # noqa: E402
from config import EVAL_DIR  # noqa: E402
from run_retrieval_eval import score_one, TOP_K  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_for_strategy(strategy: str) -> HybridRetriever:
    chunks = build_chunks(strategy)
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embedder = get_embedder()
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    return HybridRetriever(chunks, DenseIndex(chunk_ids, vectors), SparseIndex(chunk_ids, texts), embedder)


def main():
    with open(EVAL_DIR / "golden_qa.json", encoding="utf-8") as f:
        golden = [g for g in json.load(f) if g["expected_sources"]]

    print(f"n={len(golden)} answerable questions, top-{TOP_K}\n")
    print(f"{'Strategy':<18}{'Method':<10}{'Hit@' + str(TOP_K):<10}{'MRR':<8}{'#chunks'}")

    for strategy in ["fixed", "structure_aware"]:
        retriever = build_for_strategy(strategy)
        n_chunks = len(retriever._by_id)
        for method_name, fn in [
            ("dense", retriever.dense_only),
            ("sparse", retriever.sparse_only),
            ("hybrid", retriever.hybrid),
        ]:
            hits, rrs = [], []
            for item in golden:
                results = fn(item["question"], top_k=TOP_K)
                hit, rr = score_one(results, item["expected_sources"])
                hits.append(hit)
                rrs.append(rr)
            hit_rate = 100.0 * sum(hits) / len(hits)
            mrr = sum(rrs) / len(rrs)
            print(f"{strategy:<18}{method_name:<10}{hit_rate:>5.1f}%    {mrr:.3f}   {n_chunks}")


if __name__ == "__main__":
    main()
