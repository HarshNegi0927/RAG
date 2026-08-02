"""
extraction.py
Turns unstructured resume/JD text into a structured ExtractedProfile.
This is the first genuinely AI-dependent step in the pipeline: there is
no rule that reliably turns "Built and shipped a RAG pipeline with a
15-case eval harness, improved retrieval MRR from 0.83 to 1.0" into the
skills ["RAG", "Evaluation Design", "Retrieval Systems"] -- that requires
actually reading and understanding the sentence.

Uses forced tool-use (not "please respond in JSON") to get schema-
conformant structured output -- the production-relevant technique this
module demonstrates. The Pydantic schema in schemas.py IS the tool's
input_schema, so there's one source of truth for the shape of the data,
not a schema in a docstring that quietly drifts from a parser somewhere else.
"""

import json
import os
from openai import OpenAI
from .schemas import ExtractedProfile

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # Groq production model, free tier, native tool-calling

EXTRACTION_SYSTEM_PROMPT = """You extract structured information from resume or job-description text.
Only extract what is actually present or clearly implied -- do not infer skills that
aren't stated or strongly implied by described work. If years of experience isn't
stated, leave it null rather than guessing."""


def _extraction_tool():
    schema = ExtractedProfile.model_json_schema()
    return {
        "name": "record_extracted_profile",
        "description": "Record the structured profile extracted from the text.",
        "input_schema": schema,
    }


def extract_profile(text: str, source_type: str, model: str = DEFAULT_MODEL) -> ExtractedProfile:
    """source_type: 'resume' or 'job_description' -- steers extraction slightly
    (e.g. years_experience means 'candidate has' for a resume, 'required' for a JD)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. This step genuinely requires a live LLM call -- "
            "there's no rule-based fallback for reading unstructured text, by design "
            "(that's the point of this project). Get a free key at console.groq.com/keys "
            "and set it, then try again."
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    tool = _extraction_tool()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Source type: {source_type}\n\nText:\n{text}"},
        ],
        tools=[{"type": "function", "function": {
            "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"],
        }}],
        tool_choice={"type": "function", "function": {"name": tool["name"]}},
    )
    tool_call = resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return ExtractedProfile.model_validate(args)


if __name__ == "__main__":
    sample_resume = """
    Data Science student. Built a government scheme eligibility matcher using
    Python, BM25 retrieval, and the Claude API, with a 15-case evaluation harness
    (improved retrieval MRR from 0.83 to 1.0 after adding stemming). Comfortable
    with pandas, scikit-learn, and basic Streamlit deployment. 6 months of
    freelance data analysis work.
    """
    profile = extract_profile(sample_resume, source_type="resume")
    print(profile.model_dump_json(indent=2))
