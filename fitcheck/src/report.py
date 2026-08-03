"""
report.py
Final synthesis: for each partial/missing requirement, retrieve a REAL
resource from the curated knowledge base (resource_rag.py -- no LLM
involved in that lookup), then have the LLM write the human-facing
report and any suggested bullet rewrites.

The one hard rule enforced here, in the system prompt AND checked
mechanically after the call: a suggested bullet may only rephrase
experience already present in the resume. This tool is for articulating
real experience better, not for resume fraud. based_on must quote back
something that actually appears in the source resume text; if it
doesn't, the suggestion is dropped rather than shown.
"""

import json
import os
import re
from openai import OpenAI
from .schemas import ExtractedProfile, MatchReport, FullReport, BulletSuggestion, GapResource
from .resource_rag import ResourceIndex
from ._groq_strict import strict_response_format

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

REPORT_SYSTEM_PROMPT = """You write resume-improvement suggestions from a completed match analysis.

HARD RULE, no exceptions: every suggested_bullet must be a rephrasing of experience that
is ALREADY PRESENT in the candidate's resume text you're given. You are making existing
experience easier to recognize as relevant -- you are never inventing a project, tool, or
outcome the candidate didn't already describe. Set based_on to the specific resume phrase
your rewrite is grounded in; if a gap has no real resume content to rephrase, do not produce
a bullet_suggestion for it at all -- that gap should only get a gap_resource, not a fabricated bullet."""


from nltk.stem import PorterStemmer
_stemmer = PorterStemmer()

_STOPWORDS = {'a','an','the','i','my','me','to','of','in','on','for','and','or','is','are','was','were',
  'be','been','it','its','this','that','with','as','at','by','from','using','use','used','built','build'}

def _significant_tokens(text: str) -> set:
    tokens = re.findall(r"\d+\.\d+|[a-z0-9]+", text.lower())
    return {_stemmer.stem(t) if '.' not in t else t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _grounding_check(resume_text: str, based_on: str, threshold: float = 0.5) -> bool:
    """Cheap mechanical guard, independent of the LLM's own honesty: confirm the
    SIGNIFICANT terms in based_on (tools, numbers, technical nouns) actually
    appear in the source resume, so a fabricated based_on can't slip through.
    Token overlap rather than character-sequence similarity, deliberately: a
    valid rephrasing ('Applied BM25 retrieval to build a matching system') can
    share almost no character sequence with the original ('using BM25
    retrieval... matcher') while still being fully grounded. What has to
    survive a genuine rewrite is the vocabulary of tools/numbers/nouns, not
    the sentence shape."""
    resume_tokens = _significant_tokens(resume_text)
    claim_tokens = _significant_tokens(based_on)
    if not claim_tokens:
        return False
    overlap = claim_tokens & resume_tokens
    return (len(overlap) / len(claim_tokens)) >= threshold


def _report_response_format():
    schema = {
        "type": "object",
        "properties": {
            "bullet_suggestions": {"type": "array", "items": BulletSuggestion.model_json_schema()},
        },
        "required": ["bullet_suggestions"],
    }
    return strict_response_format("bullet_suggestions_report", schema)


def generate_report(resume_text: str, resume_profile: ExtractedProfile, match_report: MatchReport,
                     model: str = DEFAULT_MODEL) -> FullReport:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- report generation requires LLM reasoning, no fallback by design. Get a free key at console.groq.com/keys.")

    index = ResourceIndex()
    gap_resources: list[GapResource] = []
    for m in match_report.requirement_matches:
        if m.status in ("partial", "missing"):
            hits = index.find_resource(m.requirement + " " + m.reasoning, top_k=1)
            if hits:
                h = hits[0]
                gap_resources.append(GapResource(skill_id=h["id"], how_to_demonstrate=h["how_to_demonstrate"], practice_approach=h["practice_approach"]))

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    partials = [m for m in match_report.requirement_matches if m.status == "partial"]
    user_content = (
        f"Candidate resume text:\n{resume_text}\n\n"
        f"Requirements classified as 'partial' (candidate has SOME relevant experience, "
        f"help them state it more clearly against the JD's language):\n"
        + "\n".join(f"- {m.requirement}: {m.reasoning}" for m in partials)
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=2000,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_report_response_format(),
    )
    args = json.loads(resp.choices[0].message.content)
    raw_suggestions = [BulletSuggestion.model_validate(b) for b in args.get("bullet_suggestions", [])]

    # Mechanical anti-fabrication filter -- drop anything not grounded in the actual resume text.
    checked_suggestions = [s for s in raw_suggestions if _grounding_check(resume_text, s.based_on)]

    return FullReport(match_report=match_report, bullet_suggestions=checked_suggestions, gap_resources=gap_resources)
