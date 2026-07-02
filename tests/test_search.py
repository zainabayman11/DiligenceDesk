"""Search-provider tests — the pluggable interface, offline (no network/keys)."""

from __future__ import annotations

import json

from diligencedesk import search
from diligencedesk.tools.provider import get_tools


def test_web_search_routes_through_the_provider(monkeypatch):
    # The local web_search tool must delegate to search.run_search (the interface).
    called = {}

    def fake(query, location=None, max_results=8):
        called["query"] = query
        return [{"title": "T", "url": "https://acme-ai-health.com/x", "snippet": "s"}]

    monkeypatch.setattr(search, "run_search", fake)
    out = json.loads(get_tools("local").get("web_search").invoke({"query": "Acme overview"}))
    assert called["query"] == "Acme overview"
    assert out["results"][0]["url"] == "https://acme-ai-health.com/x"


def test_missing_key_or_provider_error_falls_back_to_ddgs(monkeypatch):
    # SEARCH_PROVIDER=tavily but no key => run_search must fall back to ddgs.
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    used = {}

    def fake_ddgs(query, location, max_results):
        used["ddgs"] = True
        return [{"title": "d", "url": "https://x.com/a", "snippet": ""}]

    monkeypatch.setattr(search, "_search_ddgs", fake_ddgs)
    results = search.run_search("PwC ETIC", location="egypt")
    assert used.get("ddgs") is True  # fell back
    assert results and results[0]["url"] == "https://x.com/a"


def test_provider_exception_also_falls_back(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "dummy")

    def boom(query, location, max_results):
        raise RuntimeError("brave api down")

    calls = {"ddgs": 0}

    def fake_ddgs(query, location, max_results):
        calls["ddgs"] += 1
        return [{"title": "d", "url": "https://x.com/a", "snippet": ""}]

    monkeypatch.setattr(search, "_search_brave", boom)
    monkeypatch.setattr(search, "_search_ddgs", fake_ddgs)
    assert search.run_search("q")  # did not raise
    assert calls["ddgs"] == 1       # fell back to ddgs


def test_dedupe_by_domain_and_path(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)  # ddgs default

    def fake_ddgs(query, location, max_results):
        return [
            {"title": "a", "url": "https://x.com/p?utm=1", "snippet": ""},
            {"title": "a2", "url": "https://www.x.com/p", "snippet": ""},  # same domain+path
            {"title": "b", "url": "https://y.com/q", "snippet": ""},
        ]

    monkeypatch.setattr(search, "_search_ddgs", fake_ddgs)
    results = search.run_search("q")
    urls = [r["url"] for r in results]
    assert len(results) == 2 and "https://y.com/q" in urls  # x.com/p collapsed to one


def test_stub_web_search_never_calls_a_real_provider(monkeypatch):
    # stub mode is fully canned: it must NOT touch the search provider layer.
    def explode(*a, **k):
        raise AssertionError("stub web_search must not call the search provider")

    monkeypatch.setattr(search, "run_search", explode)
    out = json.loads(get_tools("stub").get("web_search").invoke({"query": "Acme AI Health overview"}))
    assert out["results"]  # canned results returned, provider never called
