"""Retrieval intelligence: how we classify sources and detect signals from text.

This module holds the PURE, testable logic that turns raw search/fetch results
into graded, source-honest evidence. Keeping it separate from agents.py makes the
two most important correctness rules easy to see and unit-test:

1. EVIDENCE LEVEL IS DOMAIN-DERIVED, NEVER ASSUMED. We first resolve the company's
   official domain, then classify every source by its URL's host:
     - host is the company's own domain  -> confirmed (the company itself says it)
     - a recognised job board            -> inferred  (aspirational, not proof)
     - any other third party (news/blog/GitHub/aggregator) -> inferred
   A third-party mention can NEVER be upgraded to confirmed. (The previous version
   defaulted unknown hosts to "official_site => confirmed", which mislabelled WSJ /
   LinkedIn / blog posts as confirmed — the bug this fixes.)

2. RELEVANCE FILTERING kills noise: an open-source / engineering signal is kept
   only if it's on the company domain, under the company's own GitHub org, or the
   page clearly names the company — otherwise it's dropped.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .schemas import TechnicalSignal

_AI_CATEGORIES = ("ai_ml", "data")
_LEVEL_RANK = {"not_found": 0, "inferred": 1, "confirmed": 2}

# Recognised job boards -> evidence_type=job_posting => inferred. LinkedIn is here
# too (we classify its signals as inferred), but the jobs tool NEVER fetches it.
JOB_BOARDS = ("linkedin.com", "glassdoor.com", "wuzzuf.net", "bayt.com", "indeed.com")

# Hosts that are never a company's official site (used when resolving the domain).
NON_OFFICIAL_HOSTS = (
    "wikipedia.org", "wikimedia.org", "youtube.com", "facebook.com", "twitter.com",
    "x.com", "linkedin.com", "crunchbase.com", "bloomberg.com", "wsj.com", "reuters.com",
    "forbes.com", "techcrunch.com", "medium.com", "reddit.com", "towardsai.net",
    "glassdoor.com", "wuzzuf.net", "bayt.com", "indeed.com", "github.com", "stackoverflow.com",
)


def company_tokens(company: str) -> list[str]:
    """Distinctive lowercase tokens of a company name (used for domain/org matching)."""
    return [t for t in re.split(r"[^a-z0-9]+", company.lower()) if len(t) > 2]


def host_of(url: str) -> str:
    """The lowercased host of a URL, without a leading 'www.'."""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _host_suffix_match(host: str, domain: str) -> bool:
    """True if host is the domain or a subdomain of it (careers.x.com ~ x.com)."""
    return bool(domain) and (host == domain or host.endswith("." + domain))


def is_official(host: str, official) -> bool:
    """Is this host the company's own domain (or a subdomain of it)?"""
    if not official:
        return False
    domains = official if isinstance(official, (list, tuple, set)) else [official]
    return any(_host_suffix_match(host, d) for d in domains if d)


def is_job_board(host: str) -> bool:
    return any(_host_suffix_match(host, b) for b in JOB_BOARDS)


def resolve_official_domain(company: str, results: list[dict]) -> str | None:
    """Pick the company's official domain from search results (host, no www).

    Prefer the highest-ranked result whose host matches a company token and is not
    a known non-official host (Wikipedia, LinkedIn, news, GitHub, ...). Falls back
    to the first non-blocked host, else None.
    """
    tokens = company_tokens(company)

    def blocked(host: str) -> bool:
        # Known non-official hosts, plus generic job-aggregator TLDs (e.g. a domain
        # ending in ".jobs" is a board, never the company's own site).
        return host.endswith(".jobs") or any(_host_suffix_match(host, b) for b in NON_OFFICIAL_HOSTS)

    for r in results:
        host = host_of(r.get("url", ""))
        if not host or blocked(host):
            continue
        label = host.split(".")[0]
        if any(tok in host or tok in label for tok in tokens):
            return host
    # No host matched a company token. We DELIBERATELY do NOT fall back to a random
    # non-blocked host — guessing the wrong domain would wrongly mark third-party
    # signals as "confirmed". Returning None keeps everything honestly "inferred".
    return None


_JOBS_HOST_PREFIXES = ("jobs.", "careers.", "job.")


def _is_jobs_url(host: str, path: str) -> bool:
    """A careers/jobs URL — even on the company's OWN domain (e.g. jobs.pwc.com).

    A job posting is aspirational ("we're hiring for X"), not proof the company
    uses X, so it must be `inferred`, never `confirmed`, regardless of the host.
    """
    if host.startswith(_JOBS_HOST_PREFIXES) or ".jobs." in host:
        return True
    return "/careers" in path or "/jobs" in path or "/job/" in path or path.rstrip("/").endswith("/job")


# Location scoping: keep results that are Egypt-relevant, drop other-country pages.
# A global firm publishes country pages (jobs.us.pwc.com, pwc.com/cz/..., /uk/...);
# when the scope is Egypt we want ETIC / Middle-East / Egypt content, not those.
_EGYPT_MARKERS = ("egypt", "cairo", "etic", "middle-east", "middle east")
_FOREIGN_CC = frozenset(
    "us usa uk gb ca au in de fr es it nl cz pl jp cn sg br za ru ie ch be at se "
    "no dk fi pt gr tr mx ar cl co ph my th vn id kr hk tw nz".split()
)


def location_relevant(url: str, snippet: str, location: str) -> bool:
    """Keep a source only if it fits the location scope (default: Egypt).

    Keep when it mentions Egypt/Cairo/ETIC (or an .eg / /eg/ / /m1/ path); drop when
    it clearly belongs to another country (a foreign country-code subdomain or path
    segment); otherwise (location-neutral) keep it.
    """
    if (location or "").strip().lower() != "egypt":
        return True  # only the Egypt scope filters for now
    host = host_of(url)
    path = (urlparse(url).path or "").lower()
    text = (url + " " + (snippet or "")).lower()
    if any(m in text for m in _EGYPT_MARKERS) or host.endswith(".eg") or "/eg/" in path or "/m1/" in path:
        return True
    labels = host.split(".")
    if any(label in _FOREIGN_CC for label in labels[:-2]):  # e.g. jobs.us.pwc.com -> "us"
        return False
    segs = [s for s in path.split("/") if s]
    if segs and segs[0] in _FOREIGN_CC:  # e.g. pwc.com/cz/en/... -> "cz"
        return False
    return True  # location-neutral -> keep


def classify_evidence(url: str, official) -> tuple[str, str, str]:
    """Return (evidence_type, evidence_level, notes) for a source URL.

    The heart of the fix: the level comes from WHERE the URL points, never from an
    assumption. notes describe the REAL source so nothing is hardcoded.
    """
    host = host_of(url)
    path = (urlparse(url).path or "").lower()
    # A careers/jobs URL is a job posting FIRST — even on the official domain (a
    # US careers page like jobs.us.pwc.com/job/... is aspirational, not proof).
    if _is_jobs_url(host, path):
        return "job_posting", "inferred", "mentioned in a job posting"
    if is_official(host, official):
        if "engineering" in path or "/blog" in path or "/eng" in path:
            return "engineering_blog", "confirmed", "mentioned on the company's engineering page"
        if "/docs" in path or "developer" in path:
            return "docs", "confirmed", "mentioned in the company's docs"
        if "case" in path or "customers" in path or "success-stor" in path:
            return "case_study", "confirmed", "mentioned in a company case study"
        return "official_site", "confirmed", "mentioned on the company's official site"
    if is_job_board(host):
        return "job_posting", "inferred", "mentioned in a job posting"
    if "github.com" in host:
        return "github", "inferred", "mentioned in a GitHub repository"
    return "news", "inferred", "mentioned in a third-party source"


# --------------------------------------------------------------------------- #
# Technology catalogue + detection (domain-graded)
# --------------------------------------------------------------------------- #
# Detection is "does this exact token appear in the gathered text?", so we can
# NEVER invent a technology — only surface ones actually present in public text.
TECH_CATALOG: list[tuple[str, tuple[str, str]]] = [
    ("python", ("Python", "language")), ("javascript", ("JavaScript", "language")),
    ("typescript", ("TypeScript", "language")), ("java", ("Java", "language")),
    ("golang", ("Go", "language")), ("ruby", ("Ruby", "language")), ("php", ("PHP", "language")),
    ("scala", ("Scala", "language")), ("kotlin", ("Kotlin", "language")), ("rust", ("Rust", "language")),
    ("sql", ("SQL", "language")),
    ("fastapi", ("FastAPI", "framework")), ("django", ("Django", "framework")),
    ("flask", ("Flask", "framework")), ("spring boot", ("Spring Boot", "framework")),
    ("express", ("Express", "framework")), ("rails", ("Ruby on Rails", "framework")),
    ("langchain", ("LangChain", "framework")), ("langgraph", ("LangGraph", "framework")),
    ("react", ("React", "frontend")), ("angular", ("Angular", "frontend")),
    ("vue", ("Vue", "frontend")), ("next.js", ("Next.js", "frontend")),
    ("tailwind", ("Tailwind CSS", "frontend")),
    ("microservices", ("Microservices", "backend")), ("rest api", ("REST API", "backend")),
    ("graphql", ("GraphQL", "backend")), ("grpc", ("gRPC", "backend")), ("node.js", ("Node.js", "backend")),
    ("aws", ("AWS", "cloud")), ("azure", ("Azure", "cloud")), ("gcp", ("GCP", "cloud")),
    ("google cloud", ("Google Cloud", "cloud")),
    ("postgresql", ("PostgreSQL", "database")), ("postgres", ("PostgreSQL", "database")),
    ("mysql", ("MySQL", "database")), ("mongodb", ("MongoDB", "database")),
    ("redis", ("Redis", "database")), ("elasticsearch", ("Elasticsearch", "database")),
    ("snowflake", ("Snowflake", "database")), ("bigquery", ("BigQuery", "database")),
    ("kubernetes", ("Kubernetes", "devops")), ("docker", ("Docker", "devops")),
    ("terraform", ("Terraform", "devops")), ("jenkins", ("Jenkins", "devops")),
    ("github actions", ("GitHub Actions", "devops")), ("gitlab ci", ("GitLab CI", "devops")),
    ("ansible", ("Ansible", "devops")), ("prometheus", ("Prometheus", "devops")),
    ("grafana", ("Grafana", "devops")), ("observability", ("Observability", "devops")),
    ("odoo", ("Odoo", "erp")), ("sap", ("SAP", "erp")), ("netsuite", ("NetSuite", "erp")),
    ("salesforce", ("Salesforce", "crm")), ("hubspot", ("HubSpot", "crm")),
    ("retrieval-augmented generation", ("RAG", "ai_ml")), ("rag", ("RAG", "ai_ml")),
    ("large language model", ("LLM", "ai_ml")), ("llms", ("LLM", "ai_ml")), ("llm", ("LLM", "ai_ml")),
    ("machine learning", ("Machine Learning", "ai_ml")), ("tensorflow", ("TensorFlow", "ai_ml")),
    ("pytorch", ("PyTorch", "ai_ml")), ("hugging face", ("Hugging Face", "ai_ml")),
    ("openai", ("OpenAI", "ai_ml")), ("chatbot", ("Chatbot", "ai_ml")), ("nlp", ("NLP", "ai_ml")),
    ("computer vision", ("Computer Vision", "ai_ml")), ("mlops", ("MLOps", "ai_ml")),
    ("vector database", ("Vector Database", "ai_ml")), ("embeddings", ("Embeddings", "ai_ml")),
    ("airflow", ("Airflow", "data")), ("spark", ("Spark", "data")), ("kafka", ("Kafka", "data")),
    ("etl", ("ETL", "data")), ("data pipeline", ("Data Pipeline", "data")),
    ("data warehouse", ("Data Warehouse", "data")), ("dbt", ("dbt", "data")), ("pandas", ("pandas", "data")),
    ("workflow automation", ("Workflow Automation", "automation")), ("rpa", ("RPA", "automation")),
    ("zapier", ("Zapier", "automation")),
]


def detect_tech(hits: list[dict], official) -> list[TechnicalSignal]:
    """Find known technologies in gathered text, graded by SOURCE DOMAIN.

    Each hit is {snippet, url}. A tech on the company's own domain is confirmed;
    the same tech in a job posting / news / GitHub is inferred. We keep the
    STRONGEST level per tech — but since only the official domain yields confirmed,
    a third-party mention can never be upgraded.
    """
    found: dict[str, dict] = {}
    for h in hits:
        text = (h.get("snippet", "") or "").lower()
        if not text:
            continue
        etype, level, note = classify_evidence(h.get("url", ""), official)
        for token, (display, category) in TECH_CATALOG:
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                cur = found.get(display)
                if cur is None or _LEVEL_RANK[level] > _LEVEL_RANK[cur["level"]]:
                    found[display] = {
                        "category": category, "etype": etype, "level": level,
                        "source": h.get("url", ""), "note": note,
                    }
    signals: list[TechnicalSignal] = []
    for display, d in found.items():
        conf = {"confirmed": 0.9, "inferred": 0.5, "not_found": 0.0}[d["level"]]
        signals.append(TechnicalSignal(
            technology=display, category=d["category"], evidence_type=d["etype"],
            evidence_level=d["level"], source=d["source"], confidence=conf, notes=d["note"],
        ))
    return signals


# --------------------------------------------------------------------------- #
# Hiring-track relevance (only technical roles become tech tracks)
# --------------------------------------------------------------------------- #
TRACK_CATALOG: list[tuple[str, str]] = [
    ("ai engineer", "AI Engineer"), ("ml engineer", "AI Engineer"),
    ("machine learning engineer", "AI Engineer"), ("machine learning", "AI Engineer"),
    ("data engineer", "Data Engineer"), ("data scientist", "Data Scientist"),
    ("data analyst", "Data Analyst"),
    ("backend engineer", "Backend Engineer"), ("back-end engineer", "Backend Engineer"),
    ("frontend engineer", "Frontend Engineer"), ("full stack", "Full-Stack Engineer"),
    ("fullstack", "Full-Stack Engineer"),
    ("devops", "DevOps / Platform Engineer"), ("sre", "DevOps / Platform Engineer"),
    ("site reliability", "DevOps / Platform Engineer"), ("platform engineer", "DevOps / Platform Engineer"),
    ("cloud engineer", "DevOps / Platform Engineer"),
    ("odoo functional", "Odoo / ERP Consultant"), ("odoo developer", "Odoo / ERP Consultant"),
    ("odoo consultant", "Odoo / ERP Consultant"),
    ("erp consultant", "ERP / SAP Consultant"), ("sap consultant", "ERP / SAP Consultant"),
    ("software engineer", "Software Engineer"),
]

# Non-technical roles must never become a technical hiring track.
_NON_TECH_HINTS = (
    "hr ", "human resources", "recruit", "talent acquisition", "sales", "account manager",
    "marketing", "admin", "administrative", "reception", "finance", "accountant",
    "legal", "office manager", "procurement", "customer service", "customer support",
    "operations manager", "project coordinator", "receptionist",
)
# Generic technical hints (only used AFTER the non-technical guard).
_GENERIC_TECH_HINTS = (
    "developer", "programmer", "engineer", "architect", "software", "qa ",
    "quality assurance", "system administrator", "sysadmin",
)


def technical_track(title: str) -> str | None:
    """Map a job title to a technical track, or None if it isn't a technical role."""
    low = " " + title.lower() + " "
    for phrase, track in TRACK_CATALOG:
        if phrase in low:
            return track
    if any(nt in low for nt in _NON_TECH_HINTS):
        return None  # explicitly non-technical
    if any(k in low for k in _GENERIC_TECH_HINTS):
        return "Software Engineer"
    return None


# --------------------------------------------------------------------------- #
# Noise filter for engineering / open-source signals
# --------------------------------------------------------------------------- #
def github_org_matches(url: str, tokens: list[str]) -> bool:
    """True if a github URL's org (first path segment) belongs to the company."""
    if "github.com" not in host_of(url):
        return False
    parts = [p for p in (urlparse(url).path or "").split("/") if p]
    if not parts:
        return False
    org = parts[0].lower()
    return any(tok in org for tok in tokens)


def keep_signal(url: str, title: str, snippet: str, tokens: list[str], official) -> bool:
    """Keep an engineering/OSS item only if it's genuinely tied to the company.

    Keep if: on the company's own domain, OR a GitHub repo under the company's org,
    OR a non-GitHub page that clearly names the company. Otherwise drop it —
    unrelated repos/aggregators that merely list the name among many are removed.
    """
    host = host_of(url)
    if is_official(host, official):
        return True
    if "github.com" in host:
        return github_org_matches(url, tokens)  # ONLY the company's own org, never a list
    text = (title + " " + snippet).lower()
    return any(tok in text for tok in tokens)
