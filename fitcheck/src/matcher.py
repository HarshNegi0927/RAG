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


def _match_tool():
    # NOTE: MatchReport nests RequirementMatch, so Pydantic emits $defs/$ref in the
    # schema. Groq's function-calling (like OpenAI's) accepts $defs/$ref fine in
    # practice -- this isn't OpenAI's stricter structured-outputs mode, just normal
    # tool-calling. If a live call ever throws a schema-validation error here, the
    # fix is to inline RequirementMatch's fields directly instead of nesting the
    # model, not to change the calling code.
    return {
        "name": "record_match_report",
        "description": "Record the requirement-by-requirement match analysis.",
        "input_schema": MatchReport.model_json_schema(),
    }


def match_profiles(resume_profile: ExtractedProfile, jd_profile: ExtractedProfile, model: str = DEFAULT_MODEL) -> MatchReport:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- matching requires LLM reasoning, no rule-based fallback by design. Get a free key at console.groq.com/keys.")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    tool = _match_tool()
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
        tools=[{"type": "function", "function": {
            "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"],
        }}],
        tool_choice={"type": "function", "function": {"name": tool["name"]}},
    )
    tool_call = resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return MatchReport.model_validate(args)
