"""Structured output schemas — the project's data contracts.

WHY Pydantic models instead of free-form text:
- They are the enforcement mechanism for grounding. Specialists don't hand the
  writer a wall of prose; they hand it typed, SOURCE-CITED evidence — Claims,
  TechnicalSignals, HiringTrackSignals — and the writer assembles the brief from
  those alone. So every factual line in the report is traceable to a tool result.
- Field(description=...) on every field documents intent for a human reviewer AND
  becomes the JSON-schema description a model sees if asked to emit one.

THE EVIDENCE-LEVEL IDEA (the heart of the technical-intelligence pivot):
Tech-stack detection from public text is NOISY. A technology named in an official
engineering page is *confirmed*; the same technology listed in a job posting is
only *inferred* (they might be hiring for something new). So every technical
signal carries an explicit `evidence_level` — confirmed | inferred | not_found —
and the system NEVER presents inferred technology as confirmed, and never invents
a technology that isn't in the gathered evidence.

NOTE: there are no financial fields here. V1 is about technical intelligence
(products, tech/AI/data signals, hiring tracks, uncertainty), not financials.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# How strong the evidence is for a signal/claim. Shared vocabulary so the brief,
# the writer, and the eval can all reason about it the same way.
EvidenceLevel = Literal["confirmed", "inferred", "not_found"]


class Claim(BaseModel):
    """A single, source-cited factual statement produced by a specialist.

    The atom of evidence: a specialist never writes a sentence into the brief
    directly — it emits Claims, and the writer builds the brief out of them. A
    Claim cannot exist without a source string attached.
    """

    statement: str = Field(description="The factual statement, as a short, neutral sentence.")
    source: str = Field(
        description="Where it came from: a URL or tool id. The grounding anchor."
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Specialist confidence, 0..1."
    )
    category: str = Field(
        description="Which brief area this belongs to, e.g. overview | product | "
        "engineering | open_source."
    )
    evidence_level: EvidenceLevel = Field(
        default="confirmed",
        description="confirmed (official/eng page) | inferred (job post/news) | not_found.",
    )


class TechnicalSignal(BaseModel):
    """One detected technology, with WHERE it was found and HOW strong that is.

    `evidence_level` is derived structurally from `evidence_type` (the kind of
    source), never asserted by a model — that is how we keep 'inferred' from being
    silently upgraded to 'confirmed'.
    """

    technology: str = Field(description="The technology, e.g. 'FastAPI', 'Kubernetes', 'Odoo'.")
    category: Literal[
        "language", "framework", "cloud", "database", "devops", "erp", "crm",
        "ai_ml", "data", "frontend", "backend", "automation", "other",
    ] = Field(description="What kind of technology it is.")
    evidence_type: Literal[
        "official_site", "engineering_blog", "job_posting", "github", "docs",
        "case_study", "news", "inference", "not_found",
    ] = Field(description="The kind of source the signal came from.")
    evidence_level: EvidenceLevel = Field(
        description="Derived from evidence_type: official/eng/docs/case-study => "
        "confirmed; job posting/news/github => usually inferred."
    )
    source: str = Field(description="URL or tool id backing the signal.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="0..1, tracks evidence_level.")
    notes: str = Field(default="", description="Short context, e.g. the quoted phrase.")


class HiringTrackSignal(BaseModel):
    """A hiring direction inferred from careers/job pages (e.g. 'AI Engineer')."""

    track: str = Field(description="The role/track, e.g. 'AI Engineer', 'DevOps'.")
    evidence_type: Literal["job_posting", "career_page", "inference", "not_found"] = Field(
        description="Where the track signal came from."
    )
    source: str = Field(description="URL or tool id backing the track.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="0..1.")
    notes: str = Field(default="", description="Short context for the track.")


class RoleLens(BaseModel):
    """OPTIONAL interview-preparation lens for a target track.

    IMPORTANT: this is ANALYSIS derived from the collected signals, not a directly
    sourced factual company claim — hence the mandatory `caveat`. It only appears
    when the user supplies a target_track; the default product needs no track.
    """

    target_track: str = Field(description="The track the lens is for, e.g. 'ai_engineer'.")
    fit_summary: str = Field(description="How the company's signals align with the track.")
    skills_to_prepare: List[str] = Field(
        default_factory=list, description="Skills a candidate should brush up on."
    )
    portfolio_talking_points: List[str] = Field(
        default_factory=list, description="Project angles to highlight for this company."
    )
    interview_questions_to_prepare: List[str] = Field(
        default_factory=list, description="Likely questions to prepare answers for."
    )
    questions_to_ask_company: List[str] = Field(
        default_factory=list, description="Smart questions to ask the company."
    )
    caveat: str = Field(
        description="The mandatory caveat that this lens is interpretation, not a "
        "directly sourced company statement."
    )


class CompanyTechIntelligenceBrief(BaseModel):
    """The final deliverable: a structured, source-cited technical-intelligence brief.

    Assembled by the writer from the collected Claims / TechnicalSignals /
    HiringTrackSignals. `needs_human_review` is the honesty escape hatch: when
    coverage was thin (no tech/AI/hiring signals), the writer says so and flips
    this flag instead of inventing filler.
    """

    company_name: str = Field(description="The company the brief is about.")
    company_overview: str = Field(description="What the company does, built from claims.")
    products_and_services: List[str] = Field(
        default_factory=list, description="Products/services, each grounded in a claim."
    )
    technical_signals: List[TechnicalSignal] = Field(
        default_factory=list, description="Non-AI tech-stack signals (languages, cloud, etc.)."
    )
    ai_and_data_signals: List[TechnicalSignal] = Field(
        default_factory=list, description="AI/ML and data signals, separated out for visibility."
    )
    hiring_tracks: List[HiringTrackSignal] = Field(
        default_factory=list, description="Hiring directions from careers/job pages."
    )
    engineering_signals: List[Claim] = Field(
        default_factory=list, description="Engineering-blog / practice signals (sourced claims)."
    )
    open_source_signals: List[Claim] = Field(
        default_factory=list, description="Open-source / GitHub signals (sourced claims)."
    )
    uncertainties: List[str] = Field(
        default_factory=list,
        description="Honest notes on what was NOT found (e.g. 'no public AI signals').",
    )
    role_lens: Optional[RoleLens] = Field(
        default=None, description="Optional interview-prep lens; present only with a target_track."
    )
    sources: List[str] = Field(
        default_factory=list, description="De-duplicated list of every source cited anywhere."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence from the signals' evidence."
    )
    needs_human_review: bool = Field(
        default=False,
        description="True when coverage was thin (no technical signals found) and a "
        "human should look before relying on the brief.",
    )
    retrieval_stats: Optional[dict] = Field(
        default=None,
        description="Optional retrieval transparency: effective location, resolved "
        "official domain, pages fetched, sources by type, and jobs found per board.",
    )
