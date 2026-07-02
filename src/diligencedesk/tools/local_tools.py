"""Local-mode tools: real public data with ZERO MCP setup.

These give the agents real signals using only free, public sources and a local
file write — so the system works end-to-end with just an LLM key, no uvx/Node:

  web_search       -> ddgs (DuckDuckGo, keyless)
  fetch_page       -> httpx + a light HTML-to-text pass
  save_brief       -> writes the brief to outputs/ (also shared by stub mode)

Design rules (same as the rest of the project):
- Heavy deps (ddgs, httpx, bs4) are imported LAZILY inside each tool, so importing
  this module — e.g. just for save_brief — never requires them, and stub mode/tests
  need none of them installed.
- Tools NEVER raise. On any failure they return a JSON string with the error,
  which the specialist sees and handles, instead of crashing the graph.
- Every tool returns the SAME JSON shape as its stub-mode twin, so swapping modes
  changes no agent code.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from ..config import get_settings
from ..retrieval import host_of

_HEADERS = {"User-Agent": "DiligenceDesk/1.0 (research; +https://example.com)"}


# --------------------------------------------------------------------------- #
# web_search (routes through the pluggable provider layer — see search.py)
# --------------------------------------------------------------------------- #
@tool
def web_search(query: str) -> str:
    """Search the public web for information about a company.

    Uses whichever provider SEARCH_PROVIDER selects (ddgs | tavily | brave |
    serpapi), Egypt-scoped, with a graceful ddgs fallback. Agents are unchanged.

    Args:
        query: A search query, e.g. "Globex Industries company overview".

    Returns:
        JSON: {"results": [{"title","url","snippet"}], "query", "source"}.
    """
    try:
        from ..search import active_provider, run_search  # lazy

        results = run_search(query, max_results=8)
        return json.dumps({"results": results, "query": query, "source": active_provider()})
    except Exception as exc:  # noqa: BLE001 - never let a search failure crash the graph
        return json.dumps({"results": [], "query": query, "source": "search", "error": f"search failed: {exc}"})


# --------------------------------------------------------------------------- #
# fetch_page (httpx + light HTML->text)
# --------------------------------------------------------------------------- #
def _domain_allowed(url: str, allowed: list[str]) -> bool:
    """Return True if the URL's host is allowed (empty allowlist => allow all)."""
    if not allowed:
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:  # pragma: no cover - defensive
        return False
    return any(host == d or host.endswith("." + d) for d in allowed)


@tool
def fetch_page(url: str) -> str:
    """Fetch a web page and return its readable text (truncated).

    A read-only GET with a short timeout and an optional domain allowlist
    (ALLOWED_FETCH_DOMAINS). Use it to read a page found via web_search.

    Args:
        url: The page URL to fetch.

    Returns:
        JSON: {"url","ok","title","text","source"}. ok=false on any failure.
    """
    settings = get_settings()
    if not _domain_allowed(url, settings.allowed_fetch_domains):
        return json.dumps(
            {"url": url, "ok": False, "title": "", "text": "", "source": url,
             "error": "Domain not in ALLOWED_FETCH_DOMAINS allowlist."}
        )
    try:
        import httpx  # lazy
    except ImportError:
        return json.dumps(
            {"url": url, "ok": False, "title": "", "text": "", "source": url,
             "error": "httpx not installed (pip install httpx)."}
        )
    try:
        resp = httpx.get(
            url, timeout=settings.http_timeout, headers=_HEADERS, follow_redirects=True
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"url": url, "ok": False, "title": "", "text": "", "source": url,
             "error": f"fetch failed: {exc}"}
        )

    title, text = url, html
    try:
        from bs4 import BeautifulSoup  # lazy

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        # Drop script/style noise, then collapse to plain text.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    except Exception:  # noqa: BLE001 - if bs4 missing/fails, fall back to raw html
        pass

    # Cap the text so a huge page can't blow up the LLM context or our state.
    text = text[:4000]
    return json.dumps({"url": url, "ok": True, "title": title, "text": text, "source": url})


# --------------------------------------------------------------------------- #
# find_jobs (Egypt-scoped hiring signals from four job boards)
# --------------------------------------------------------------------------- #
# (source_name, site-scope domain, do_fetch). LinkedIn is search-result ONLY: its
# pages sit behind a login wall, so we NEVER fetch a linkedin.com URL — we take the
# title + URL straight from the search-result snippet.
_JOB_SOURCES = [
    ("glassdoor", "glassdoor.com", True),
    ("linkedin", "linkedin.com/jobs", False),
    ("wuzzuf", "wuzzuf.net", True),
    ("bayt", "bayt.com", True),
]
# Location terms added to every jobs query, on top of the provider's region param.
_LOCATION_TERMS = {"egypt": "Egypt Cairo"}


def _fetch_title(url: str) -> str:
    """Fetch a job listing and return a cleaner title, or '' — NEVER for LinkedIn.

    The hard LinkedIn-safety guard lives here too (belt and suspenders): even if a
    caller asked, a linkedin.com URL is never fetched.
    """
    if "linkedin.com" in host_of(url):
        return ""
    try:
        import httpx

        resp = httpx.get(url, timeout=get_settings().http_timeout, headers=_HEADERS, follow_redirects=True)
        resp.raise_for_status()
        from bs4 import BeautifulSoup

        title = BeautifulSoup(resp.text, "html.parser").title
        return title.string.strip()[:120] if title and title.string else ""
    except Exception:  # noqa: BLE001 - best-effort; fall back to the search title
        return ""


@tool
def find_jobs(company: str, location: str = "egypt", track: str = "") -> str:
    """Find job postings for a company, scoped to a location (default: Egypt).

    Aggregates from Glassdoor, LinkedIn (search-result only), Wuzzuf, and Bayt. The
    LinkedIn listing page is NEVER opened (login wall) — only its search snippet.

    Args:
        company: The company to search jobs for.
        location: Location scope, e.g. "egypt". Egypt/Cairo terms are added.
        track: Optional role hint to narrow the search, e.g. "ai engineer".

    Returns:
        JSON: {"results": [{"title","company","location","url","source"}], "location", "source"}.
    """
    from ..search import active_provider, run_search  # lazy

    loc_terms = _LOCATION_TERMS.get(location.lower(), location)
    # Quote only the BRAND (first 1-2 words) for precision, e.g. `"PwC ETIC"`.
    # Quoting a long legal name ("PwC ETIC PricewaterhouseCoopers") as an exact
    # phrase matches almost no job posting, which returned zero jobs.
    brand = " ".join(company.split()[:2])
    company_q = f'"{brand}"' if brand else ""
    postings: list[dict] = []
    seen: set[str] = set()
    for source, domain, do_fetch in _JOB_SOURCES:
        query = " ".join(x for x in [f"site:{domain}", company_q, track, loc_terms] if x).strip()
        try:
            results = run_search(query, location=location, max_results=5)
        except Exception:  # noqa: BLE001 - a search failure for one board shouldn't kill the rest
            results = []
        for r in results[:3]:
            url = r.get("url", "")
            title = r.get("title", "")
            if not url or url in seen:
                continue
            seen.add(url)
            if do_fetch:  # never true for LinkedIn
                refined = _fetch_title(url)
                if refined:
                    title = refined
            postings.append({"title": title, "company": company, "location": location, "url": url, "source": source})
    return json.dumps({"results": postings, "location": location, "source": f"{active_provider()}-jobs"})


# --------------------------------------------------------------------------- #
# save_brief (local file write — shared by stub + local modes)
# --------------------------------------------------------------------------- #
def _outputs_dir() -> Path:
    """Resolve <project_root>/outputs, independent of the current directory."""
    # local_tools.py -> tools -> diligencedesk -> src -> project root
    outputs = Path(__file__).resolve().parents[3] / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return outputs


def _slugify(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "brief"


@tool
def save_brief(content: str, filename: str = "") -> str:
    """Save a finished technical-intelligence brief to a Markdown file in outputs/.

    A read-only-to-the-web ACTION: it only writes a local file under the
    project's outputs/ folder; it never sends data anywhere.

    Args:
        content: The brief text (Markdown).
        filename: Optional name; a timestamped default is used if empty.

    Returns:
        JSON: {"ok","path","source"}; ok=false with an error on failure.
    """
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = _slugify(filename) if filename else "brief"
        path = _outputs_dir() / f"{base}_{stamp}.md"
        path.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path), "source": "local-fs"})
    except OSError as exc:
        return json.dumps({"ok": False, "path": "", "source": "local-fs", "error": str(exc)})


# The named tools local mode exposes (financials are out of scope in the V1
# technical-intelligence pivot).
LOCAL_TOOLS = [web_search, fetch_page, find_jobs, save_brief]
