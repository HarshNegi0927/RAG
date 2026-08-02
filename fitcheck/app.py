"""
Streamlit demo for FitCheck AI.
Run with: streamlit run app.py
Requires GROQ_API_KEY (free, no credit card -- console.groq.com/keys) --
this project's core value is the LLM reasoning, so unlike a RAG-only
project there's no meaningful fallback mode here (see README for why
that's a deliberate choice, not a gap).
"""

import os
import streamlit as st
from RAG.fitcheck.src.graph import build_graph, LOW_FIT_THRESHOLD

st.set_page_config(page_title="FitCheck AI", page_icon="🎯", layout="centered")
st.title("🎯 FitCheck AI")
st.caption(
    "Paste your resume and a target job description. An LLM reads both, reasons about "
    "requirement-by-requirement fit, and suggests bullet rewrites -- grounded only in "
    "experience you actually have, never invented."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Unlike the eligibility-filter project, this one has "
        "no rule-based fallback -- reading and reasoning about unstructured resume/JD text "
        "is the actual point, and there's no honest way to fake that without an LLM. "
        "Get a free key (no credit card) at console.groq.com/keys, then set it and restart:\n\n"
        "`$env:GROQ_API_KEY=\"gsk_...\"` (PowerShell) or `export GROQ_API_KEY=gsk_...` (bash/zsh)."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Your resume (paste as plain text)", height=280,
                                placeholder="Data Science student. Built a government scheme eligibility matcher...")
with col2:
    jd_text = st.text_area("Target job description", height=280,
                            placeholder="Looking for a Data Scientist with 3+ years...")

if st.button("Analyze fit", type="primary", disabled=not (resume_text and jd_text)):
    with st.spinner("Extracting profiles, matching requirements, and writing the report..."):
        graph = build_graph()
        result = graph.invoke({
            "resume_text": resume_text, "jd_text": jd_text,
            "resume_profile": None, "jd_profile": None,
            "match_report": None, "full_report": None, "low_fit_note": None,
        })

    if result.get("low_fit_note"):
        st.warning(result["low_fit_note"])
        st.caption(f"(Routed here because fit score was below {LOW_FIT_THRESHOLD}/100 -- see graph.py's conditional edge.)")
    else:
        report = result["full_report"]
        mr = report.match_report
        st.metric("Overall fit score", f"{mr.overall_fit_score}/100")
        st.write(mr.fit_summary)

        st.subheader("Requirement-by-requirement breakdown")
        for m in mr.requirement_matches:
            icon = {"matched": "✅", "partial": "🟡", "missing": "🔴"}[m.status]
            with st.expander(f"{icon} {m.requirement}"):
                st.write(f"**Reasoning:** {m.reasoning}")
                if m.evidence:
                    st.write(f"**Evidence from your resume:** {m.evidence}")

        if report.bullet_suggestions:
            st.subheader("Suggested bullet rewrites")
            st.caption("Each one passed a mechanical check confirming it's grounded in something you actually wrote -- not invented.")
            for b in report.bullet_suggestions:
                st.markdown(f"**For:** {b.gap_or_partial}")
                st.code(b.suggested_bullet, language=None)
                st.caption(f"Based on: \"{b.based_on}\"")

        if report.gap_resources:
            st.subheader("Closing the remaining gaps")
            for g in report.gap_resources:
                st.markdown(f"**{g.skill_id.replace('_', ' ').title()}**")
                st.write(g.how_to_demonstrate)
                st.caption(g.practice_approach)
