"""Tool-layer tests — the uniform interface, and tools that never raise."""

from __future__ import annotations

import json

import pytest

from diligencedesk.tools.provider import REQUIRED_TOOLS, Toolbox, get_tools


def _names(tb: Toolbox) -> set[str]:
    return set(tb.names())


def test_required_tools_are_search_fetch_jobs_save():
    # find_jobs was added for Egypt-scoped hiring signals; financials stay removed.
    assert set(REQUIRED_TOOLS) == {"web_search", "fetch_page", "find_jobs", "save_brief"}


def test_stub_toolbox_exposes_required_tools():
    assert _names(get_tools("stub")) >= set(REQUIRED_TOOLS)


def test_local_toolbox_builds_without_network():
    assert _names(get_tools("local")) >= set(REQUIRED_TOOLS)


def test_mcp_mode_falls_back_to_local_when_no_servers(monkeypatch):
    # If the MCP servers can't be started, mcp mode must still return a WORKING
    # toolbox (resilience), not crash. We simulate the failure so the test never
    # actually spawns uvx/npx (which would hit the network).
    from diligencedesk.tools import provider

    async def _boom(_settings):
        raise RuntimeError("simulated: no MCP servers available")

    monkeypatch.setattr(provider, "_load_mcp_tools", _boom)
    assert _names(get_tools("mcp")) >= set(REQUIRED_TOOLS)


def test_toolbox_get_unknown_raises():
    with pytest.raises(KeyError):
        get_tools("stub").get("does_not_exist")


def test_stub_web_search_is_parseable_and_company_aware():
    tb = get_tools("stub")
    out = json.loads(tb.get("web_search").invoke({"query": "Acme AI Health technology stack"}))
    assert out["results"], "known company should return canned results"
    empty = json.loads(tb.get("web_search").invoke({"query": "Nonexistent LLC overview"}))
    assert empty["results"] == []


def test_stub_web_search_routes_by_query_bucket():
    # The same company returns DIFFERENT results depending on the query intent.
    tb = get_tools("stub")
    tech = json.loads(tb.get("web_search").invoke(
        {"query": "Acme AI Health site:acme-ai-health.com engineering blog technology"}))
    prod = json.loads(tb.get("web_search").invoke(
        {"query": "Acme AI Health site:acme-ai-health.com products solutions"}))
    assert any("engineering" in r["url"] or "docs" in r["url"] for r in tech["results"])
    assert any("products" in r["url"] or "solutions" in r["url"] for r in prod["results"])


def test_stub_fetch_page_handles_unknown_url_gracefully():
    out = json.loads(get_tools("stub").get("fetch_page").invoke({"url": "https://nope.example/x"}))
    assert out["ok"] is False and out["text"] == ""
