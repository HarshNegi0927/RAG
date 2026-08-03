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
    check them against without calling the model. This runs real Groq
    calls against eval/llm_steps_tests.json (hand-labeled resume/JD pairs
    with a directional expectation: weak fit should score low and show
    real 'missing' requirements, strong fit should score high), rather
    than exact-string matching -- LLM wording varies run to run, so the
    check is on the signal that actually matters (skills present, fit
    score direction), not on reproducing identical text.
    """
    if not os.environ.get("GROQ_API_KEY"):
        print("--- LLM-only steps (extraction, matching) ---")
        print("  Skipped: no GROQ_API_KEY in this environment. Not faked -- run this with a key to get real numbers.\n")
        return None

    from RAG.fitcheck.src.extraction import extract_profile
    from RAG.fitcheck.src.matcher import match_profiles

    tests = json.load(open(os.path.join(os.path.dirname(__file__), "llm_steps_tests.json"), encoding="utf-8"))
    correct = 0
    print("--- LLM-only steps (extraction, matching) -- real Groq calls ---")
    for t in tests:
        try:
            resume_profile = extract_profile(t["resume_text"], "resume")
            jd_profile = extract_profile(t["jd_text"], "job_description")
            match = match_profiles(resume_profile, jd_profile)
        except Exception as e:
            print(f"  [ERROR] {t['id']}: call failed -- {e}")
            continue

        got_skills = {s.lower() for s in resume_profile.skills}
        skills_ok = any(k in " ".join(got_skills) for k in t["expect_extracted_skills_include_any"])

        score = match.overall_fit_score
        if "expect_fit_score_max" in t:
            score_ok = score <= t["expect_fit_score_max"]
            direction = f"<= {t['expect_fit_score_max']}"
        else:
            score_ok = score >= t["expect_fit_score_min"]
            direction = f">= {t['expect_fit_score_min']}"

        ok = skills_ok and score_ok
        correct += ok
        print(f"  [{'OK' if ok else 'CHECK'}] {t['id']}: skills_found={skills_ok}, "
              f"fit_score={score} (expected {direction})")
        if not ok:
            print(f"         extracted skills: {sorted(got_skills)}")
            print(f"         fit summary: {match.fit_summary}")

    print(f"  {correct}/{len(tests)} within expected range\n")
    return correct, len(tests)


if __name__ == "__main__":
    r_hits, r_total = eval_resource_retrieval()
    g_hits, g_total = eval_grounding_check()
    llm_result = eval_llm_steps()

    total_hits, total_count = r_hits + g_hits, r_total + g_total
    if llm_result is not None:
        l_hits, l_total = llm_result
        total_hits += l_hits
        total_count += l_total

    print("=" * 60)
    label = "ALL" if llm_result is not None else "testable without a key"
    print(f"TOTAL ({label}): {total_hits}/{total_count}")
    print("=" * 60)
