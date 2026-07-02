"""Config tests — env-driven, per-role models, safe defaults."""

from __future__ import annotations

import pytest

from diligencedesk import config
from diligencedesk.config import default_model, get_settings


def test_defaults_are_offline_and_sane(monkeypatch):
    # With only the conftest defaults, we get stub everything + a finite turn cap.
    monkeypatch.delenv("MAX_SUPERVISOR_TURNS", raising=False)
    s = get_settings()
    assert s.tool_mode == "stub"
    assert s.role("supervisor").provider == "stub"
    assert s.max_supervisor_turns == 12  # the cost/safety cap


def test_per_role_providers_and_models(monkeypatch):
    # The headline SLM lever: the supervisor and workers can run on DIFFERENT
    # providers/models, configured purely via env.
    monkeypatch.setenv("SUPERVISOR_PROVIDER", "groq")
    monkeypatch.setenv("WORKER_PROVIDER", "gemini")
    s = get_settings()
    assert s.role("supervisor").provider == "groq"
    assert s.role("supervisor").model == "llama-3.3-70b-versatile"  # stronger router
    assert s.role("worker").provider == "gemini"
    assert s.role("worker").model == "gemini-2.5-flash-lite"  # smaller SLM
    # An explicit model override wins over the per-(role,provider) default.
    monkeypatch.setenv("WORKER_MODEL", "my-custom-slm")
    assert get_settings().role("worker").model == "my-custom-slm"


def test_unknown_tool_mode_raises(monkeypatch):
    monkeypatch.setenv("TOOL_MODE", "telepathy")
    with pytest.raises(ValueError):
        get_settings()


def test_bad_int_env_falls_back(monkeypatch):
    # An operator typo must not crash startup — we use the safe default.
    monkeypatch.setenv("MAX_SUPERVISOR_TURNS", "twelve")
    assert get_settings().max_supervisor_turns == 12


def test_default_model_table():
    assert default_model("worker", "groq") == "llama-3.1-8b-instant"
    assert default_model("supervisor", "groq") == "llama-3.3-70b-versatile"
    assert default_model("worker", "stub") == "stub"


def test_responsible_ai_policy_states_golden_rule():
    # The policy string is a real control — it must forbid inventing facts and
    # require honest confirmed-vs-inferred evidence.
    policy = config.RESPONSIBLE_AI_POLICY.lower()
    assert "never invent" in policy
    assert "inferred" in policy and "confirmed" in policy
