"""
evaluate.py
Runs everything in this pipeline that can be evaluated without a live
GROQ_API_KEY: resource retrieval (RAG layer) and the anti-fabrication
grounding check. Extraction and matching are genuinely LLM-only steps by
design (see extraction.py's docstring) -- eval_llm_steps() below shows
exactly how to eval them once a key is available; it is not run by default.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from RAG.fitcheck.src.resource_rag import ResourceIndex
from RAG.fitcheck.src.report import _grounding_check


def eval_resource_retrieval():
    idx = ResourceIndex()
    tests = json.load(open(os.path.join(os.path.dirname(__file__), "resource_retrieval_tests.json"), encoding="utf-8"))
    hits = 0
    print("--- Resource retrieval (RAG layer) ---")
    for t in tests:
        results = idx.find_resource(t["gap"], top_k=1)
        got = results[0]["id"] if results else "NONE"
        ok = got in t["expect_top2"]
        hits += ok
        print(f"  [{'OK' if ok else 'MISS'}] {t['id']}: {got}")
    print(f"  {hits}/{len(tests)} correct top-1 retrieval\n")
    return hits, len(tests)


def eval_grounding_check():
    data = json.load(open(os.path.join(os.path.dirname(__file__), "grounding_check_tests.json"), encoding="utf-8"))
    resume = data["resume_text"]
    correct = 0
    print("--- Anti-fabrication grounding check ---")
    for c in data["cases"]:
        got = _grounding_check(resume, c["claim"])
        ok = got == c["expect"]
        correct += ok
        label = "grounded" if c["expect"] else "fabricated"
        print(f"  [{'OK' if ok else 'WRONG'}] expected={label:10s} got={got}: {c['claim'][:55]}")
    print(f"  {correct}/{len(data['cases'])} correct\n")
    return correct, len(data["cases"])


def eval_llm_steps():
    """Extraction (unstructured text -> structured profile) and matching
    (JD requirement vs resume evidence, with reasoning) are the two steps
    that are genuinely LLM-only -- there is no offline ground truth to
    check them against without calling the model. To eval them properly
    once you have a key:
      1. Hand-label 10-15 (resume, jd) pairs with the skills a human
         reviewer would extract from each, and the fit classification
         a human reviewer would give each requirement.
      2. Run extract_profile() and match_profiles() on the same inputs.
      3. Score extraction as set-overlap (precision/recall on skills)
         against your hand labels, same methodology as the scheme-navigator
         eligibility eval.
      4. Score matching by agreement rate against your hand labels on
         matched/partial/missing, and spot-check the `reasoning` field
         for whether it's actually pointing at real evidence.
    This function is a stub, not a fabricated result -- there's no
    number here in this environment because doing so would mean either
    calling a key that isn't available, or making up numbers, and both
    are worse than saying plainly that this needs to be run for real.
    """
    if not os.environ.get("GROQ_API_KEY"):
        print("--- LLM-only steps (extraction, matching) ---")
        print("  Skipped: no GROQ_API_KEY in this environment. Not faked -- see docstring for how to run this for real.\n")
        return None
    # left intentionally unimplemented in this environment -- wire this up locally with your key.
    raise NotImplementedError("Implement against your own hand-labeled test pairs once you have a key.")


if __name__ == "__main__":
    r_hits, r_total = eval_resource_retrieval()
    g_hits, g_total = eval_grounding_check()
    eval_llm_steps()
    print("=" * 60)
    print(f"TOTAL (testable without a key): {r_hits + g_hits}/{r_total + g_total}")
    print("=" * 60)
