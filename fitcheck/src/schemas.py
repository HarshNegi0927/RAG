"""
schemas.py
Pydantic models the LLM extraction/matching/report steps are constrained
to return. Structured output (vs. free-form text you regex out of a
response) is the actual production-relevant skill being demonstrated
here -- it's what makes the pipeline's later steps reliable enough to
chain in a graph instead of parsing prose.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ExtractedProfile(BaseModel):
    """What extraction.py turns raw resume OR job-description text into."""
    skills: list[str] = Field(description="Concrete skills/tools/technologies mentioned, normalized to common names (e.g. 'PyTorch' not 'pytorch framework').")
    years_experience: float | None = Field(default=None, description="Total relevant years of experience, if inferable. Null if not stated/not applicable (e.g. a JD).")
    role_level: Literal["entry", "mid", "senior", "unspecified"] = "unspecified"
    project_highlights: list[str] = Field(default_factory=list, description="Specific projects or achievements mentioned, kept close to the source wording.")
    domains: list[str] = Field(default_factory=list, description="Application domains present, e.g. 'healthcare', 'fintech', 'NLP'.")


class RequirementMatch(BaseModel):
    """One JD requirement, matched against the resume."""
    requirement: str
    status: Literal["matched", "partial", "missing"]
    evidence: str = Field(description="Direct evidence from the resume if matched/partial; empty string if missing.")
    reasoning: str = Field(description="Why this was classified this way -- required even for 'matched', so the classification is auditable.")


class MatchReport(BaseModel):
    requirement_matches: list[RequirementMatch]
    overall_fit_score: int = Field(ge=0, le=100)
    fit_summary: str = Field(description="2-3 sentence honest summary of overall fit, including real weaknesses.")


class BulletSuggestion(BaseModel):
    gap_or_partial: str = Field(description="Which requirement this addresses.")
    suggested_bullet: str = Field(description="A rewritten resume bullet using ONLY experience already present in the resume -- never fabricated.")
    based_on: str = Field(description="The exact resume content this rewrite is grounded in, so it's checkable against fabrication.")


class GapResource(BaseModel):
    skill_id: str
    how_to_demonstrate: str
    practice_approach: str


class FullReport(BaseModel):
    match_report: MatchReport
    bullet_suggestions: list[BulletSuggestion]
    gap_resources: list[GapResource]
