"""
Proof-of-value script: same queries, three retrieval modes, side by side.

This is deliberately not a unit test -- it's meant to be read. Each query
below was picked to stress one side of hybrid search:

  - exact_identifier queries contain a specific term (an env var name, a
    function name) that BM25 should nail and pure dense search might not
    rank as highly, because word vectors smear a rare identifier across
    its semantic neighborhood instead of matching it exactly.

  - paraphrase queries deliberately avoid the document's exact wording,
    to see whether dense search finds the right chunk via meaning when
    BM25 has little to no token overlap to work with.

Run: python3 scripts/compare_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_index import build_retriever  # noqa: E402

TEST_QUERIES = [
    {
        "query": "What environment variable controls the database migration timeout?",
        "expects_source": "database-migration-guide.md",
        "kind": "exact_identifier (DB_MIGRATION_TIMEOUT_SEC)",
    },
    {
        "query": "Who should I contact if something breaks in the middle of the night?",
        "expects_source": "incident-response-runbook.md",
        "kind": "paraphrase (no 'on-call' / 'SEV1' / 'incident' wording used)",
    },
    {
        "query": "How do I get a new API key if the old one might be compromised?",
        "expects_source": "api-authentication-guide.md",
        "kind": "paraphrase (real doc says 'rotate_api_key', not 'get a new key')",
    },
    {
        "query": "How long is a customer's data kept after they delete their account?",
        "expects_source": "data-retention-privacy-policy.md",
        "kind": "paraphrase (real doc says 'closes their account' / 'retained for 90 days')",
    },
    {
        "query": "What services are involved when a customer gets billed?",
        "expects_source": "service-architecture-overview.md",
        "kind": "semantic (billing-service, ledger-service, payment.events.v2)",
    },
]


def rank_of(results, expects_source: str) -> str:
    for r in results:
        if r.chunk.source_file == expects_source:
            return f"#{r.rank}"
    return "not in top-k"


def main():
    print("Building retriever (structure-aware chunking, local embedder)...\n")
    retriever, chunks = build_retriever()
    print(f"Indexed {len(chunks)} chunks from {len({c.source_file for c in chunks})} docs.\n")
    print("=" * 100)

    for case in TEST_QUERIES:
        q = case["query"]
        dense = retriever.dense_only(q, top_k=5)
        sparse = retriever.sparse_only(q, top_k=5)
        hybrid = retriever.hybrid(q, top_k=5)

        print(f"\nQUERY: {q}")
        print(f"Stress-testing: {case['kind']}")
        print(f"Correct source doc: {case['expects_source']}")
        print("-" * 100)
        print(f"{'Method':<10} {'Rank of correct doc':<22} {'Top result (source :: heading)'}")
        for label, results in [("dense", dense), ("sparse", sparse), ("hybrid", hybrid)]:
            top = results[0] if results else None
            top_desc = f"{top.chunk.source_file} :: {top.chunk.section_heading or '(no heading)'}" if top else "(none)"
            print(f"{label:<10} {rank_of(results, case['expects_source']):<22} {top_desc}")
        print("=" * 100)


if __name__ == "__main__":
    main()
