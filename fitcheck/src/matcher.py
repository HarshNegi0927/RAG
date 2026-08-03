"""
matcher.py
The actual reasoning step: given both extracted profiles, decide which JD
requirements are matched / partial / missing, WITH reasoning and evidence
for every single classification -- including the "matched" ones, so a
human reviewer can audit why the model believed a match, not just trust
a label. This is the step that can't be done by keyword overlap: "led
a cross-functional analytics initiative" satisfying "stakeholder
management experience" requires understanding that one implies the
other, not that they share vocabulary.
"""

import json
import os
from openai import OpenAI
from .schemas import ExtractedProfile, MatchReport
from ._groq_strict import strict_response_format

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

MATCH_SYSTEM_PROMPT = """You compare a candidate's extracted profile against a job description's
extracted requirements and classify the fit honestly.

Rules:
- For EVERY requirement in the JD profile's skills/domains, classify it: matched, partial, or missing.
- "matched" requires real evidence in the candidate profile, not vibes -- if you can't point to
  specific evidence, it's "partial" at best.
- Give reasoning for every classification, including matches -- this must be auditable.
- Do not be generous to make the candidate look better. An honest low fit score is more useful
  to them than a flattering inaccurate one.
- overall_fit_score should reflect: matched requirements weigh fully, partial requirements weigh
  about half, missing requirements weigh zero, roughly proportional to the number of requirements."""


def _match_response_format():
    return strict_response_format("match_report", MatchReport.model_json_schema())


def match_profiles(resume_profile: ExtractedProfile, jd_profile: ExtractedProfile, model: str = DEFAULT_MODEL) -> MatchReport:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- matching requires LLM reasoning, no rule-based fallback by design. Get a free key at console.groq.com/keys.")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    user_content = (
        f"Candidate profile:\n{resume_profile.model_dump_json(indent=2)}\n\n"
        f"Job requirements profile:\n{jd_profile.model_dump_json(indent=2)}\n\n"
        "Classify every requirement in the job profile's skills and domains."
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=2000,
        messages=[
            {"role": "system", "content": MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_match_response_format(),
    )
    args = json.loads(resp.choices[0].message.content)
    return MatchReport.model_validate(args)
