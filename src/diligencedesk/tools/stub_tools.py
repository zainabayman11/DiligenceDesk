"""Canned, fully-offline tools + a fixed corpus of synthetic companies.

WHY a fixed synthetic corpus: behaviour and evaluation must be STABLE. Each company
has a resolvable OFFICIAL DOMAIN plus a deliberate MIX of sources so we can exercise
the correctness rules offline:
  - official-domain pages  -> confirmed signals
  - third-party pages (WSJ, towardsai)  -> must stay INFERRED (never confirmed)
  - the company's own GitHub org  -> kept; an unrelated "awesome-list" repo -> dropped
  - Egypt job postings across boards, mixing technical and non-technical titles.

All facts live in the fixtures; the agents do the detection/classification, so the
same fixtures drive the CLI demo, the tests, AND the eval. Tools never raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.tools import tool

from .local_tools import save_brief  # noqa: F401  (re-exported into the toolbox)


@dataclass
class CompanyFixture:
    """All canned signals for one synthetic company, keyed by search bucket."""

    name: str
    official_domain: str
    overview_hits: list[dict] = field(default_factory=list)
    product_hits: list[dict] = field(default_factory=list)
    tech_hits: list[dict] = field(default_factory=list)
    ai_hits: list[dict] = field(default_factory=list)
    engineering_hits: list[dict] = field(default_factory=list)
    jobs: list[dict] = field(default_factory=list)
    pages: dict[str, str] = field(default_factory=dict)


_BUILTIN: list[CompanyFixture] = [
    CompanyFixture(
        name="Acme AI Health",
        official_domain="acme-ai-health.com",
        overview_hits=[
            {"title": "Acme AI Health - About", "url": "https://acme-ai-health.com/about",
             "snippet": "Acme AI Health builds AI-powered clinical decision-support software for hospitals and clinics."},
            {"title": "Acme AI Health - Wikipedia", "url": "https://en.wikipedia.org/wiki/Acme_AI_Health",
             "snippet": "Acme AI Health is a health-technology company."},
        ],
        product_hits=[
            {"title": "Products - Acme AI Health", "url": "https://acme-ai-health.com/products",
             "snippet": "Acme AI Health's products include a clinical chatbot, a RAG-based medical knowledge assistant, and a patient triage platform."},
            {"title": "Solutions - Acme", "url": "https://acme-ai-health.com/solutions",
             "snippet": "Acme offers an AI documentation assistant and analytics dashboards for clinics."},
        ],
        tech_hits=[
            {"title": "Engineering - Acme", "url": "https://acme-ai-health.com/engineering",
             "snippet": "Our core stack is Python, FastAPI, and PostgreSQL, deployed on AWS."},
            # THIRD-PARTY tech mention -> must stay INFERRED (would FAIL the old logic).
            {"title": "Acme AI Health - WSJ", "url": "https://www.wsj.com/articles/acme-ai-health-tech",
             "snippet": "Acme reportedly uses Kubernetes and TensorFlow, according to people familiar."},
        ],
        ai_hits=[
            {"title": "AI at Acme", "url": "https://acme-ai-health.com/engineering/ai",
             "snippet": "Acme AI Health uses Retrieval-Augmented Generation (RAG) and LLMs, backed by a vector database."},
            {"title": "Acme ML - towardsai.net", "url": "https://towardsai.net/p/acme-ml",
             "snippet": "Acme applies machine learning and NLP for clinical notes."},
        ],
        engineering_hits=[
            {"title": "Acme engineering blog", "url": "https://acme-ai-health.com/engineering/blog/rag-pipeline",
             "snippet": "Building a HIPAA-compliant RAG pipeline at Acme AI Health."},
            {"title": "acme-ai-health/clinical-nlp", "url": "https://github.com/acme-ai-health/clinical-nlp",
             "snippet": "Acme AI Health open-source clinical NLP tools."},
            # Unrelated repo that merely LISTS the company name -> must be DROPPED.
            {"title": "awesome-health-ai", "url": "https://github.com/someone/awesome-health-ai",
             "snippet": "A curated list of health-AI companies including Acme AI Health and many others."},
        ],
        jobs=[
            {"title": "Machine Learning Engineer", "url": "https://www.glassdoor.com/job-listing/mle-acme-egypt", "source": "glassdoor", "location": "Cairo, Egypt"},
            {"title": "Data Engineer", "url": "https://www.linkedin.com/jobs/view/acme-data-engineer-egypt", "source": "linkedin", "location": "Egypt"},
            {"title": "Backend Developer", "url": "https://wuzzuf.net/jobs/p/acme-backend-developer-cairo-egypt", "source": "wuzzuf", "location": "Cairo, Egypt"},
            {"title": "HR Manager", "url": "https://www.bayt.com/en/egypt/jobs/acme-hr-manager", "source": "bayt", "location": "Cairo, Egypt"},
        ],
        pages={
            "https://acme-ai-health.com/about": "Acme AI Health builds AI-powered clinical decision-support software. Its business model is SaaS subscriptions sold to hospitals and clinics.",
            "https://acme-ai-health.com/products": "Acme AI Health's products include a clinical chatbot, a RAG-based medical knowledge assistant, and a patient triage platform.",
            "https://acme-ai-health.com/engineering": "Our core stack is Python, FastAPI, and PostgreSQL, deployed on AWS.",
            "https://acme-ai-health.com/engineering/ai": "Acme AI Health uses Retrieval-Augmented Generation (RAG) and LLMs, backed by a vector database.",
        },
    ),
    CompanyFixture(
        name="Globex Cloud Systems",
        official_domain="globex-cloud.com",
        overview_hits=[
            {"title": "Globex Cloud Systems - About", "url": "https://globex-cloud.com/about",
             "snippet": "Globex Cloud Systems provides large-scale backend and cloud infrastructure services for enterprises."},
            {"title": "Globex - TechCrunch", "url": "https://techcrunch.com/globex-cloud-systems",
             "snippet": "Globex Cloud Systems, an infrastructure provider, expanded its platform."},
        ],
        product_hits=[
            {"title": "Products - Globex", "url": "https://globex-cloud.com/products",
             "snippet": "Globex Cloud Systems offers managed Kubernetes, observability, and microservices platforms."},
        ],
        tech_hits=[
            {"title": "Engineering - Globex", "url": "https://globex-cloud.com/engineering",
             "snippet": "Globex Cloud Systems is built on Java, Spring Boot, Kubernetes, and AWS, following a microservices architecture."},
            {"title": "Globex API Docs", "url": "https://globex-cloud.com/docs/api",
             "snippet": "Globex services expose REST API and gRPC interfaces."},
        ],
        ai_hits=[],  # no public AI -> the no-AI honesty path
        engineering_hits=[
            {"title": "Globex engineering blog", "url": "https://globex-cloud.com/engineering/blog/k8s-scale",
             "snippet": "Running Kubernetes at scale with observability via Prometheus and Grafana at Globex."},
            {"title": "globex-cloud/terraform-modules", "url": "https://github.com/globex-cloud/terraform-modules",
             "snippet": "Globex Cloud Systems open-source Terraform modules."},
        ],
        jobs=[
            {"title": "Backend Engineer", "url": "https://www.glassdoor.com/job-listing/backend-globex-egypt", "source": "glassdoor", "location": "Cairo, Egypt"},
            {"title": "DevOps Engineer", "url": "https://wuzzuf.net/jobs/p/globex-devops-cairo", "source": "wuzzuf", "location": "Cairo, Egypt"},
            {"title": "Cloud Engineer", "url": "https://www.linkedin.com/jobs/view/globex-cloud-engineer-egypt", "source": "linkedin", "location": "Egypt"},
            {"title": "Sales Manager", "url": "https://www.bayt.com/en/egypt/jobs/globex-sales-manager", "source": "bayt", "location": "Cairo, Egypt"},
        ],
        pages={
            "https://globex-cloud.com/about": "Globex Cloud Systems provides backend and cloud infrastructure. Business model: usage-based managed services and enterprise support contracts.",
            "https://globex-cloud.com/products": "Globex Cloud Systems offers managed Kubernetes, observability, and microservices platforms.",
            "https://globex-cloud.com/engineering": "Globex Cloud Systems is built on Java, Spring Boot, Kubernetes, and AWS, following a microservices architecture.",
            "https://globex-cloud.com/docs/api": "Globex services expose REST API and gRPC interfaces.",
        },
    ),
    CompanyFixture(
        name="Initech ERP Services",
        official_domain="initech-erp.com",
        overview_hits=[
            {"title": "Initech ERP Services - About", "url": "https://initech-erp.com/about",
             "snippet": "Initech ERP Services implements and customizes ERP systems for mid-market businesses."},
        ],
        product_hits=[
            {"title": "Services - Initech", "url": "https://initech-erp.com/services",
             "snippet": "Initech offers Odoo implementation, SAP integration, and workflow automation services."},
        ],
        tech_hits=[
            {"title": "Technology - Initech", "url": "https://initech-erp.com/technology",
             "snippet": "Initech ERP Services specializes in Odoo and Python customization with PostgreSQL backends and workflow automation."},
            {"title": "Initech Case Study", "url": "https://initech-erp.com/customers/manufacturing",
             "snippet": "An Initech case study describes integrating Odoo with SAP for a manufacturing client."},
        ],
        ai_hits=[],
        engineering_hits=[
            {"title": "initech-erp/odoo-modules", "url": "https://github.com/initech-erp/odoo-modules",
             "snippet": "Initech open-source Odoo modules."},
        ],
        jobs=[
            {"title": "Odoo Functional Consultant", "url": "https://wuzzuf.net/jobs/p/initech-odoo-cairo", "source": "wuzzuf", "location": "Cairo, Egypt"},
            {"title": "ERP Consultant", "url": "https://www.bayt.com/en/egypt/jobs/initech-erp-consultant", "source": "bayt", "location": "Cairo, Egypt"},
            {"title": "Accountant", "url": "https://www.glassdoor.com/job-listing/initech-accountant-egypt", "source": "glassdoor", "location": "Cairo, Egypt"},
        ],
        pages={
            "https://initech-erp.com/about": "Initech ERP Services implements ERP systems. Business model: consulting and implementation billed per project.",
            "https://initech-erp.com/services": "Initech offers Odoo implementation, SAP integration, and workflow automation services.",
            "https://initech-erp.com/technology": "Initech ERP Services specializes in Odoo and Python customization with PostgreSQL backends and workflow automation.",
        },
    ),
    CompanyFixture(
        name="NoSignal Consulting",
        official_domain="nosignal-consulting.com",
        overview_hits=[
            {"title": "NoSignal Consulting - About", "url": "https://nosignal-consulting.com/about",
             "snippet": "NoSignal Consulting is a general business consulting firm."},
        ],
        product_hits=[],
        tech_hits=[],
        ai_hits=[],
        engineering_hits=[],
        jobs=[],
        pages={"https://nosignal-consulting.com/about": "NoSignal Consulting provides general business consulting services. Limited public technical information is available."},
    ),
]

_CORPUS: dict[str, CompanyFixture] = {fx.name.lower(): fx for fx in _BUILTIN}


def builtin_company_names() -> list[str]:
    return [fx.name for fx in _BUILTIN]


def reset_corpus() -> None:
    global _CORPUS
    _CORPUS = {fx.name.lower(): fx for fx in _BUILTIN}


def register_company(fixture: CompanyFixture) -> None:
    _CORPUS[fixture.name.lower()] = fixture


def fixture_for(name: str) -> CompanyFixture | None:
    """Look up a fixture by company name (used by the eval to derive allowed sources)."""
    return _CORPUS.get(name.lower())


def all_source_urls(fx: CompanyFixture) -> set[str]:
    """Every URL the tools could legitimately emit for a fixture (for groundedness)."""
    urls: set[str] = set(fx.pages.keys())
    for bucket in (fx.overview_hits, fx.product_hits, fx.tech_hits, fx.ai_hits, fx.engineering_hits):
        urls.update(h.get("url", "") for h in bucket if h.get("url"))
    urls.update(j.get("url", "") for j in fx.jobs if j.get("url"))
    return urls


def _match_company(query: str) -> CompanyFixture | None:
    """Find the fixture whose name appears in the query (longest name wins)."""
    q = query.lower()
    for name in sorted(_CORPUS, key=len, reverse=True):
        if name in q:
            return _CORPUS[name]
    return None


def _bucket_for_query(query: str) -> str:
    """Route a query to the right canned bucket by keyword (mirrors the agents).

    Order matters. We check tech/AI markers BEFORE products, and use "product"/
    "solution" (NOT the generic "service") for the products bucket, so a company
    name that contains a common word (e.g. "Initech ERP Services") doesn't get
    mis-routed to products on its tech query.
    """
    q = query.lower()
    if any(w in q for w in ("github", "open source", "open-source", "repositor")):
        return "engineering"
    if any(w in q for w in ("machine learning", "artificial intelligence", "data platform")):
        return "ai"
    if any(w in q for w in ("technology", "tech stack", "framework", "programming", "engineering", "blog", " stack")):
        return "tech"
    if any(w in q for w in ("product", "solution")):
        return "products"
    return "overview"


@tool
def web_search(query: str) -> str:
    """Search the public web for information about a company.

    Returns up to a handful of results, each with a title, url, and snippet.

    Args:
        query: A search query, e.g. "Acme AI Health technology stack".

    Returns:
        JSON: {"results": [{"title","url","snippet"}], "query", "source"}.
    """
    fx = _match_company(query)
    if fx is None:
        return json.dumps({"results": [], "query": query, "source": "stub", "note": "Company not in the stub corpus."})
    bucket = _bucket_for_query(query)
    hits = {
        "overview": fx.overview_hits, "products": fx.product_hits, "tech": fx.tech_hits,
        "ai": fx.ai_hits, "engineering": fx.engineering_hits,
    }[bucket]
    return json.dumps({"results": hits, "query": query, "source": "stub"})


@tool
def fetch_page(url: str) -> str:
    """Fetch and return the readable text of a web page.

    Args:
        url: The page URL (typically one returned by web_search).

    Returns:
        JSON: {"url","ok","title","text","source"}. ok is false on a miss; never raises.
    """
    for fx in _CORPUS.values():
        if url in fx.pages:
            return json.dumps({"url": url, "ok": True, "title": fx.name, "text": fx.pages[url], "source": url})
    return json.dumps({"url": url, "ok": False, "title": "", "text": "", "source": url,
                       "error": "No canned page for this URL in the stub corpus."})


@tool
def find_jobs(company: str, location: str = "egypt", track: str = "") -> str:
    """Find job postings for a company, scoped to a location (default: Egypt).

    Stub version: returns the company's canned Egypt postings across job boards.

    Args:
        company: The company to search jobs for.
        location: Location scope (kept for parity; the stub postings are Egypt-scoped).
        track: Optional role hint (ignored by the stub; the agent filters relevance).

    Returns:
        JSON: {"results": [{"title","company","location","url","source"}], "location", "source"}.
    """
    fx = _match_company(company)
    jobs = []
    if fx:
        jobs = [{**j, "company": fx.name} for j in fx.jobs]
    return json.dumps({"results": jobs, "location": location, "source": "stub-jobs"})


STUB_TOOLS = [web_search, fetch_page, find_jobs, save_brief]
