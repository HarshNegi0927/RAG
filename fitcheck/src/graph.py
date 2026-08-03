"""
graph.py
LangGraph orchestration for the full pipeline. This is deliberately NOT
a linear chain dressed up in graph syntax -- a plain sequence of function
calls would do that job with no need for LangGraph at all. The reason a
graph earns its place here is the conditional branch after matching: a
genuinely poor fit (score < 25) doesn't get the same "here's how to
close the gap" report as a borderline one -- it gets routed to a
different node that says so plainly instead of forcing bullet-point
optimism onto a fundamentally mismatched application.

    START -> extract_resume -> extract_jd -> match --+-- (score < 25) --> low_fit_advisory -> END
                                                       +-- (score >= 25) --> generate_report -> END
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from .extraction import extract_profile
from .matcher import match_profiles
from .report import generate_report
from .schemas import ExtractedProfile, MatchReport, FullReport

LOW_FIT_THRESHOLD = 25


class PipelineState(TypedDict):
    resume_text: str
    jd_text: str
    resume_profile: Optional[ExtractedProfile]
    jd_profile: Optional[ExtractedProfile]
    match_report: Optional[MatchReport]
    full_report: Optional[FullReport]
    low_fit_note: Optional[str]


def extract_resume_node(state: PipelineState) -> dict:
    return {"resume_profile": extract_profile(state["resume_text"], "resume")}


def extract_jd_node(state: PipelineState) -> dict:
    return {"jd_profile": extract_profile(state["jd_text"], "job_description")}


def match_node(state: PipelineState) -> dict:
    return {"match_report": match_profiles(state["resume_profile"], state["jd_profile"])}


def route_on_fit(state: PipelineState) -> str:
    score = state["match_report"].overall_fit_score
    return "low_fit_advisory" if score < LOW_FIT_THRESHOLD else "generate_report"


def report_node(state: PipelineState) -> dict:
    full = generate_report(state["resume_text"], state["resume_profile"], state["match_report"])
    return {"full_report": full}


def low_fit_advisory_node(state: PipelineState) -> dict:
    score = state["match_report"].overall_fit_score
    missing = [m.requirement for m in state["match_report"].requirement_matches if m.status == "missing"]
    note = (
        f"Fit score is {score}/100 -- below the threshold where bullet-rewrite suggestions "
        f"would be honest advice. {len(missing)} core requirements aren't met at all "
        f"({', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}). "
        "Tailoring language won't close a gap this size. Worth treating this as a "
        "longer-term target role and looking for a closer-fit role to apply to now."
    )
    return {"low_fit_note": note}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("extract_resume", extract_resume_node)
    g.add_node("extract_jd", extract_jd_node)
    g.add_node("match", match_node)
    g.add_node("generate_report", report_node)
    g.add_node("low_fit_advisory", low_fit_advisory_node)

    g.add_edge(START, "extract_resume")
    g.add_edge("extract_resume", "extract_jd")
    g.add_edge("extract_jd", "match")
    g.add_conditional_edges("match", route_on_fit, {"low_fit_advisory": "low_fit_advisory", "generate_report": "generate_report"})
    g.add_edge("generate_report", END)
    g.add_edge("low_fit_advisory", END)
    return g.compile()


if __name__ == "__main__":
    import os
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set -- this runs the real graph, which needs a live key.")
        print("Graph structure compiles fine though, checking that much:")
        graph = build_graph()
        print(graph.get_graph().draw_mermaid())
    else:
        graph = build_graph()
        result = graph.invoke({
            "resume_text": "Data Science student. Built a government scheme eligibility matcher "
                            "using Python, BM25 retrieval, and the Claude API, with a 15-case "
                            "evaluation harness. Comfortable with pandas, scikit-learn, basic Streamlit.",
            "jd_text": "Looking for a Data Scientist with 3+ years production ML experience, "
                       "strong SQL, deep learning (PyTorch), and experience deploying models at scale.",
            "resume_profile": None, "jd_profile": None, "match_report": None,
            "full_report": None, "low_fit_note": None,
        })
        print(result)
