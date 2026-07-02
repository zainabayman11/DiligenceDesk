"""Pluggable web-search providers behind ONE interface.

WHY: the free `ddgs` (DuckDuckGo) backend is weak and rate-limits, which made
results shallow and jobs empty. So search is now a small provider interface —
`run_search(query, location)` — selected by the `SEARCH_PROVIDER` env var. The
local `web_search` / `find_jobs` tools call `run_search`; the AGENTS are unchanged.

Providers (all return the uniform shape `{title, url, snippet}`):
  ddgs    default, free, keyless (offline/dev fallback).
  tavily  free tier, AI-focused           -> TAVILY_API_KEY
  brave   free tier                        -> BRAVE_API_KEY
  serpapi real Google results (optional)   -> SERPAPI_API_KEY

Rules baked in here:
- Egypt locale by default: each provider passes its own region param
  (SerpAPI gl=eg&hl=en, Brave country=EG, Tavily country=egypt), driven by
  `location` (from the --location flag / DILIGENCE_LOCATION).
- GRACEFUL FALLBACK: if the chosen provider errors or its key is missing, we fall
  back to ddgs — a run never fails because a paid provider hiccuped.
- Results are de-duplicated by domain+path.
- stub mode never reaches here (its canned web_search is separate), so tests stay
  offline and deterministic.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

_HEADERS = {"User-Agent": "DiligenceDesk/1.0 (research)"}

# Per-provider region codes for the supported locations.
_DDGS_REGION = {"egypt": "eg-en"}
_BRAVE_CC = {"egypt": "EG"}
_SERP_GL = {"egypt": "eg"}
_TAVILY_COUNTRY = {"egypt": "egypt"}

_REAL_PROVIDERS = ("tavily", "brave", "serpapi")


_PROVIDER_KEYS = {"tavily": "TAVILY_API_KEY", "brave": "BRAVE_API_KEY", "serpapi": "SERPAPI_API_KEY"}


def active_provider() -> str:
    """The provider that will actually be used (for the trace/retrieval_stats).

    Honest about fallback: a real provider with no key always falls back to ddgs,
    so we label it "<provider> (no key -> ddgs)" rather than pretending it ran.
    """
    p = os.getenv("SEARCH_PROVIDER", "ddgs").strip().lower() or "ddgs"
    if p in _REAL_PROVIDERS and not os.getenv(_PROVIDER_KEYS[p]):
        return f"{p} (no key -> ddgs)"
    return p


def _location() -> str:
    return (os.getenv("DILIGENCE_LOCATION", "egypt") or "egypt").strip().lower()


def _uniform(items, title_k, url_k, snip_k) -> list[dict]:
    out = []
    for x in items or []:
        url = x.get(url_k) or x.get("href") or x.get("url") or ""
        out.append({"title": x.get(title_k, ""), "url": url, "snippet": x.get(snip_k, "") or ""})
    return out


def _dedupe(results: list[dict]) -> list[dict]:
    """De-duplicate by domain+path (so ?utm=... variants collapse to one)."""
    seen, out = set(), []
    for r in results:
        url = r.get("url", "")
        if not url:
            continue
        p = urlparse(url)
        key = (p.hostname or "").lower().lstrip("www.") + (p.path or "").rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Providers — each returns a list[dict], or None when its key is missing.
# --------------------------------------------------------------------------- #
def _search_ddgs(query: str, location: str, max_results: int) -> list[dict]:
    from ddgs import DDGS  # lazy

    region = _DDGS_REGION.get(location)
    with DDGS() as ddgs:
        try:
            raw = ddgs.text(query, max_results=max_results, region=region) if region else ddgs.text(query, max_results=max_results)
        except TypeError:  # older/newer ddgs without region kwarg
            raw = ddgs.text(query, max_results=max_results)
    return _uniform(raw, "title", "href", "body")


def _search_tavily(query: str, location: str, max_results: int) -> list[dict] | None:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return None
    import httpx  # lazy

    body = {"api_key": key, "query": query, "max_results": max_results, "search_depth": "basic"}
    country = _TAVILY_COUNTRY.get(location)
    if country:
        body["country"] = country
    resp = httpx.post("https://api.tavily.com/search", json=body, timeout=25, headers=_HEADERS)
    resp.raise_for_status()
    return _uniform(resp.json().get("results", []), "title", "url", "content")


def _search_brave(query: str, location: str, max_results: int) -> list[dict] | None:
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return None
    import httpx  # lazy

    params = {"q": query, "count": max_results}
    cc = _BRAVE_CC.get(location)
    if cc:
        params["country"] = cc
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params=params, timeout=25,
        headers={**_HEADERS, "X-Subscription-Token": key, "Accept": "application/json"},
    )
    resp.raise_for_status()
    return _uniform(resp.json().get("web", {}).get("results", []), "title", "url", "description")


def _search_serpapi(query: str, location: str, max_results: int) -> list[dict] | None:
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        return None
    import httpx  # lazy

    params = {"engine": "google", "q": query, "api_key": key, "num": max_results}
    gl = _SERP_GL.get(location)
    if gl:
        params.update({"gl": gl, "hl": "en"})
    resp = httpx.get("https://serpapi.com/search", params=params, timeout=30, headers=_HEADERS)
    resp.raise_for_status()
    return _uniform(resp.json().get("organic_results", []), "title", "link", "snippet")


_PROVIDERS = {"tavily": _search_tavily, "brave": _search_brave, "serpapi": _search_serpapi}


def run_search(query: str, location: str | None = None, max_results: int = 8) -> list[dict]:
    """Search via the configured provider, with a graceful ddgs fallback.

    Args:
        query: the search query.
        location: locale scope (defaults to DILIGENCE_LOCATION / "egypt").
        max_results: how many results to request.

    Returns:
        De-duplicated `[{title, url, snippet}]`. Never raises.
    """
    provider = active_provider()
    loc = (location or _location()).lower()

    results = None
    if provider in _REAL_PROVIDERS:
        try:
            results = _PROVIDERS[provider](query, loc, max_results)  # None if key missing
        except Exception:  # noqa: BLE001 - any provider error -> fall back to ddgs
            results = None
    if not results:  # ddgs default, OR fallback when a real provider was empty/failed
        try:
            results = _search_ddgs(query, loc, max_results)
        except Exception:  # noqa: BLE001 - even ddgs failing must not crash the graph
            results = []
    return _dedupe(results or [])
