"""Jobs-tool tests — Egypt-scoped aggregation + the hard LinkedIn-never-fetched rule."""

from __future__ import annotations

import json

from diligencedesk import search
from diligencedesk.tools import local_tools
from diligencedesk.tools.provider import get_tools


def test_stub_find_jobs_returns_egypt_postings_from_multiple_sources():
    tb = get_tools("stub")
    out = json.loads(tb.get("find_jobs").invoke({"company": "Acme AI Health", "location": "egypt", "track": ""}))
    results = out["results"]
    assert results, "stub should return canned Egypt postings"
    sources = {j["source"] for j in results}
    assert {"glassdoor", "linkedin", "wuzzuf", "bayt"} <= sources  # all four boards
    assert all("egypt" in j["location"].lower() for j in results)


def test_local_find_jobs_never_fetches_a_linkedin_url(monkeypatch):
    """The jobs tool may fetch Glassdoor/Wuzzuf/Bayt, but NEVER a linkedin.com page."""
    fetched: list[str] = []

    def fake_search(query, location=None, max_results=8):
        if "linkedin" in query:
            return [{"title": "Data Engineer", "url": "https://www.linkedin.com/jobs/view/1", "snippet": ""}]
        if "glassdoor" in query:
            return [{"title": "ML Engineer", "url": "https://www.glassdoor.com/job/1", "snippet": ""}]
        if "wuzzuf" in query:
            return [{"title": "Backend Engineer", "url": "https://wuzzuf.net/jobs/1", "snippet": ""}]
        if "bayt" in query:
            return [{"title": "DevOps Engineer", "url": "https://www.bayt.com/jobs/1", "snippet": ""}]
        return []

    def spy_fetch(url):
        fetched.append(url)
        return ""

    monkeypatch.setattr(search, "run_search", fake_search)  # jobs route through the provider layer
    monkeypatch.setattr(local_tools, "_fetch_title", spy_fetch)

    out = json.loads(local_tools.find_jobs.invoke({"company": "Acme", "location": "egypt", "track": ""}))

    # Postings were aggregated across boards...
    assert out["results"]
    # ...LinkedIn appears (from the search snippet)...
    assert any("linkedin.com" in local_tools.host_of(j["url"]) for j in out["results"])
    # ...but the tool NEVER fetched a linkedin.com URL.
    assert fetched, "the fetchable boards should have been fetched"
    assert all("linkedin.com" not in local_tools.host_of(u) for u in fetched)
