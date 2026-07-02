"""Schema contract tests — the data contracts must enforce grounding + evidence levels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diligencedesk.schemas import (
    Claim,
    CompanyTechIntelligenceBrief,
    HiringTrackSignal,
    RoleLens,
    TechnicalSignal,
)


def test_claim_requires_source_and_has_evidence_level():
    c = Claim(statement="X does Y.", source="https://example.com", category="overview")
    assert c.confidence == 0.7 and c.evidence_level == "confirmed"  # defaults
    with pytest.raises(ValidationError):
        Claim(statement="x", source="s", category="overview", confidence=1.5)  # out of range
    with pytest.raises(ValidationError):
        Claim(statement="x", source="s", category="overview", evidence_level="maybe")  # bad enum


def test_technical_signal_constrains_category_and_levels():
    s = TechnicalSignal(
        technology="FastAPI", category="framework", evidence_type="official_site",
        evidence_level="confirmed", source="https://x",
    )
    assert s.confidence == 0.5  # default
    with pytest.raises(ValidationError):
        TechnicalSignal(technology="x", category="banana", evidence_type="official_site",
                        evidence_level="confirmed", source="s")
    with pytest.raises(ValidationError):
        TechnicalSignal(technology="x", category="framework", evidence_type="rumor",
                        evidence_level="confirmed", source="s")


def test_hiring_track_signal_defaults():
    h = HiringTrackSignal(track="AI Engineer", evidence_type="job_posting", source="https://x")
    assert h.confidence == 0.5
    with pytest.raises(ValidationError):
        HiringTrackSignal(track="x", evidence_type="telepathy", source="s")


def test_role_lens_requires_caveat():
    rl = RoleLens(target_track="ai_engineer", fit_summary="...", caveat="interpretation, not fact")
    assert rl.skills_to_prepare == [] and rl.interview_questions_to_prepare == []
    with pytest.raises(ValidationError):
        RoleLens(target_track="ai_engineer", fit_summary="...")  # missing caveat


def test_brief_round_trips_and_role_lens_optional():
    brief = CompanyTechIntelligenceBrief(
        company_name="Acme AI Health",
        company_overview="Acme builds clinical AI software.",
        products_and_services=["clinical chatbot"],
        technical_signals=[TechnicalSignal(technology="Python", category="language",
                                           evidence_type="official_site", evidence_level="confirmed", source="https://x")],
        ai_and_data_signals=[TechnicalSignal(technology="RAG", category="ai_ml",
                                             evidence_type="engineering_blog", evidence_level="confirmed", source="https://x")],
        hiring_tracks=[HiringTrackSignal(track="AI Engineer", evidence_type="career_page", source="https://x")],
        uncertainties=["No public open-source signals were found."],
        sources=["https://x"], confidence=0.8,
    )
    again = CompanyTechIntelligenceBrief.model_validate(brief.model_dump())
    assert again.company_name == "Acme AI Health"
    assert again.role_lens is None  # optional, defaults to None
    assert again.needs_human_review is False
    assert again.retrieval_stats is None  # optional transparency field


def test_brief_accepts_optional_retrieval_stats():
    brief = CompanyTechIntelligenceBrief(
        company_name="Acme AI Health", company_overview="x",
        retrieval_stats={"location": "egypt", "official_domain": "acme-ai-health.com", "pages_fetched": 4},
    )
    dumped = CompanyTechIntelligenceBrief.model_validate(brief.model_dump())
    assert dumped.retrieval_stats["official_domain"] == "acme-ai-health.com"
