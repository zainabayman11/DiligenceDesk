"""Technical-signals agent tests — domain-graded evidence, jobs, filtering, coverage."""

from __future__ import annotations

from diligencedesk.agents import technical_signals_agent
from diligencedesk.config import get_settings
from diligencedesk.state import initial_state
from diligencedesk.tools.provider import get_tools


def _run(company: str, official: str | None = None):
    settings = get_settings()
    toolbox = get_tools("stub", settings)
    state = initial_state(company, None, "egypt")
    if official:
        state["official_domain"] = official
    return technical_signals_agent(state, toolbox, settings)


def _levels(signals) -> dict[str, str]:
    return {s.technology: s.evidence_level for s in signals}


def test_evidence_levels_are_domain_derived():
    upd = _run("Acme AI Health", "acme-ai-health.com")
    levels = _levels(upd["technical_signals"])
    assert levels.get("Python") == "confirmed"      # on the official engineering page
    assert levels.get("Kubernetes") == "inferred"   # only in a third-party (WSJ) article


def test_no_invented_tech_and_no_ai_honesty():
    upd = _run("Globex Cloud Systems", "globex-cloud.com")
    techs = {s.technology for s in upd["technical_signals"]}
    assert "RAG" not in techs and "LLM" not in techs   # not present -> not invented
    ai = [s for s in upd["technical_signals"] if s.category in ("ai_ml", "data")]
    assert ai == []
    assert any("No AI or data" in u for u in upd["uncertainties"])


def test_hiring_tracks_are_technical_only():
    upd = _run("Acme AI Health", "acme-ai-health.com")
    tracks = {h.track for h in upd["hiring_tracks"]}
    assert "AI Engineer" in tracks and "Data Engineer" in tracks
    # The "HR Manager" posting must NOT become a technical hiring track.
    assert not any(t.lower().startswith("hr") or "manager" == t.lower() for t in tracks)
    assert all(h.source for h in upd["hiring_tracks"])  # every track is sourced


def test_noise_repo_dropped_from_open_source():
    upd = _run("Acme AI Health", "acme-ai-health.com")
    oss = [c.source for c in upd["claims"] if c.category == "open_source"]
    assert any("clinical-nlp" in s for s in oss)          # company's own org kept
    assert not any("awesome-health-ai" in s for s in oss)  # unrelated repo dropped


def test_coverage_notes_for_empty_sections():
    upd = _run("NoSignal Consulting", "nosignal-consulting.com")
    unc = " | ".join(upd["uncertainties"])
    assert "No tech-stack signals" in unc
    assert "No AI or data" in unc
    assert "No public hiring-track" in unc


def test_hiring_present_means_no_no_hiring_line():
    # If any board returned a posting, we must NOT emit the "no hiring signals" line.
    upd = _run("Acme AI Health", "acme-ai-health.com")
    assert upd["hiring_tracks"]
    assert not any("No public hiring-track" in u for u in upd["uncertainties"])
