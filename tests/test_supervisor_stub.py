"""Integration tests for the supervisor + specialists, driven by the stub.

These prove the whole multi-agent loop works WITHOUT a real model or network:
- the supervisor routes to both specialists, then the writer;
- specialists emit grounded claims/signals into shared state;
- the writer builds a VALID CompanyTechIntelligenceBrief;
- the loop TERMINATES (and the turn cap is a hard backstop);
- the optional role lens behaves: absent without a track, populated with one.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from diligencedesk.config import get_settings
from diligencedesk.graph import build_graph
from diligencedesk.schemas import CompanyTechIntelligenceBrief
from diligencedesk.state import initial_state


def _run(company: str, thread: str = "t", target_track=None, settings=None):
    settings = settings or get_settings()
    app = build_graph(settings=settings)
    config = {"configurable": {"thread_id": thread}}
    app.invoke(initial_state(company, target_track), config=config)
    return app.get_state(config).values


def test_supervisor_routes_to_both_specialists_then_writer():
    state = _run("Acme AI Health")
    assert set(state["visited"]) == {"company_research_agent", "technical_signals_agent"}
    brief = state["brief"]
    assert isinstance(brief, CompanyTechIntelligenceBrief)
    CompanyTechIntelligenceBrief.model_validate(brief.model_dump())


def test_specialists_emit_grounded_evidence():
    state = _run("Acme AI Health")
    # Company research produced sourced claims...
    assert state["claims"] and all(c.source for c in state["claims"])
    # ...and technical signals are all sourced with valid evidence levels.
    sigs = state["technical_signals"]
    assert sigs and all(s.source and s.evidence_level in ("confirmed", "inferred", "not_found") for s in sigs)
    assert state["hiring_tracks"]


def test_confirmed_signals_are_on_the_official_domain():
    # THE correctness rule: a confirmed signal's source must be the company's own
    # domain — a third-party mention can never be confirmed.
    from diligencedesk.retrieval import host_of, is_official

    brief = _run("Acme AI Health")["brief"]
    official = brief.retrieval_stats["official_domain"]
    assert official == "acme-ai-health.com"
    all_sig = list(brief.technical_signals) + list(brief.ai_and_data_signals)
    for s in all_sig:
        if s.evidence_level == "confirmed":
            assert is_official(host_of(s.source), official), f"{s.technology} confirmed off-domain"
    # And a third-party-only tech (WSJ) stayed inferred.
    kub = {s.technology: s.evidence_level for s in all_sig}
    assert kub.get("Kubernetes") == "inferred"


def test_brief_buckets_signals_and_fills_products():
    brief = _run("Acme AI Health")["brief"]
    # AI techs land in ai_and_data_signals, not the general technical_signals list.
    assert any(s.technology == "RAG" for s in brief.ai_and_data_signals)
    assert all(s.category not in ("ai_ml", "data") for s in brief.technical_signals)
    # products_and_services is filled from the official products page.
    assert brief.products_and_services
    # retrieval transparency is recorded.
    assert brief.retrieval_stats and brief.retrieval_stats.get("location") == "egypt"


def test_no_signal_company_flags_review_and_uncertainty():
    brief = _run("NoSignal Consulting")["brief"]
    assert brief.technical_signals == [] and brief.hiring_tracks == []
    assert brief.needs_human_review is True
    assert any("tech-stack" in u for u in brief.uncertainties)


def test_role_lens_absent_without_track_present_with_track():
    # Default product: no track => no role lens.
    assert _run("Acme AI Health", thread="nolens")["brief"].role_lens is None
    # With a track => a populated lens. Robust contract: skills_to_prepare AND
    # interview_questions_to_prepare are non-empty (other lists are best-effort).
    rl = _run("Acme AI Health", thread="lens", target_track="ai_engineer")["brief"].role_lens
    assert rl is not None
    assert len(rl.skills_to_prepare) > 0 and len(rl.interview_questions_to_prepare) > 0
    assert "interpretation" in rl.caveat.lower()  # marked as analysis, not fact


def test_run_is_deterministic():
    a = _run("Globex Cloud Systems", thread="a")
    b = _run("Globex Cloud Systems", thread="b")
    assert len(a["claims"]) == len(b["claims"])
    assert {s.technology for s in a["technical_signals"]} == {s.technology for s in b["technical_signals"]}


def test_turn_cap_forces_termination(monkeypatch):
    monkeypatch.setenv("MAX_SUPERVISOR_TURNS", "1")
    settings = get_settings()
    state = _run("Acme AI Health", thread="cap", settings=settings)
    brief = state["brief"]
    assert isinstance(brief, CompanyTechIntelligenceBrief)
    # The cap hit before both specialists ran -> honest review flag.
    assert len(state["visited"]) < 2
    assert brief.needs_human_review is True


def test_routing_survives_an_llm_error():
    # A routing-LLM failure (e.g. 429 quota) must NOT crash the run — the supervisor
    # falls back to the first pending specialist (deterministic order).
    from diligencedesk import supervisor

    def boom(role, settings=None):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    orig = supervisor.build_llm
    supervisor.build_llm = boom
    try:
        choice = supervisor._route_with_llm(
            get_settings(), "X", [], ["company_research_agent", "technical_signals_agent"])
        assert choice == "company_research_agent"
    finally:
        supervisor.build_llm = orig


def test_status_messages_are_compact():
    state = _run("Acme AI Health")
    ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
    assert ai and all(len(str(m.content)) < 200 for m in ai)
