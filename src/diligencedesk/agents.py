"""The specialist agents: company-research, technical-signals, and writer.

THE CORE DESIGN — structured, DOMAIN-GRADED evidence, not free writing:
Each specialist gathers evidence via the toolbox and contributes typed,
SOURCE-CITED objects to shared state. Two correctness rules drive this version:

1. Resolve the company's OFFICIAL DOMAIN first, then grade every signal by the
   source's host (see retrieval.py): the company's own pages are `confirmed`;
   job boards / news / third-party / GitHub are `inferred`. A third-party mention
   is NEVER upgraded to confirmed.

2. Retrieval goes DEEPER and is EGYPT-SCOPED: targeted site-scoped queries fetch
   official product/engineering pages, and a dedicated jobs tool aggregates Egypt
   postings across four boards. Noise (unrelated repos, aggregators) is filtered
   out; coverage gaps are stated honestly.

The writer assembles the brief from the collected objects ALONE and records
lightweight retrieval stats for transparency.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import RESPONSIBLE_AI_POLICY, Settings
from .llm import build_llm
from .retrieval import (
    _AI_CATEGORIES,
    classify_evidence,
    company_tokens,
    detect_tech,
    host_of,
    is_job_board,
    is_official,
    keep_signal,
    location_relevant,
    resolve_official_domain,
    technical_track,
)

# Extra query terms that bias site-scoped searches toward a location.
_LOCATION_QUERY_TERMS = {"egypt": "Egypt Cairo"}


def _loc_terms(location: str) -> str:
    return _LOCATION_QUERY_TERMS.get((location or "").strip().lower(), location or "")
from .schemas import (
    Claim,
    CompanyTechIntelligenceBrief,
    HiringTrackSignal,
    RoleLens,
)
from .state import DiligenceState
from .stub_llm import CLAIMS_MARKER
from .tools.provider import Toolbox

ROLE_LENS_CAVEAT = (
    "This role lens is an interpretation based on collected public signals, not a "
    "directly sourced company statement."
)


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _parse_json(text: str):
    """Tolerantly parse JSON the LLM returned (handles ``` fences / stray prose)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _tool(toolbox: Toolbox, name: str, **kwargs):
    """Invoke a toolbox tool by name and parse its JSON result (None on garbage)."""
    return _parse_json(toolbox.get(name).invoke(kwargs))


def _search(toolbox: Toolbox, query: str) -> list[dict]:
    """Run web_search and return its results list (empty on miss)."""
    return (_tool(toolbox, "web_search", query=query) or {}).get("results", [])


def _evidence_for(stub: dict, evidence: list[dict]) -> dict | None:
    """Map a claim stub back to its evidence item by index (the grounding anchor)."""
    try:
        idx = int(stub.get("i"))
    except (TypeError, ValueError):
        return None
    return evidence[idx] if 0 <= idx < len(evidence) else None


def _extract_claims(llm, role_instruction: str, company: str, evidence: list[dict]):
    """Ask the worker LLM to turn EVIDENCE into grounded claim stubs (by index)."""
    if not evidence:
        return []
    system = SystemMessage(content=RESPONSIBLE_AI_POLICY + "\n" + role_instruction)
    human = HumanMessage(
        content=(
            f"{CLAIMS_MARKER}\n"
            f"Company: {company}\n"
            "From the EVIDENCE below, extract only claims the evidence supports. "
            "Cite each by its evidence index 'i'. Do not add anything not present.\n"
            f"<EVIDENCE>{json.dumps(evidence)}</EVIDENCE>\n"
            'Reply with ONLY a JSON array of objects: '
            '{"i": <int>, "statement": <one sentence>, "confidence": <0..1>}.'
        )
    )
    parsed = _parse_json(str(llm.invoke([system, human]).content))
    return parsed if isinstance(parsed, list) else []


def _status(name: str, text: str) -> AIMessage:
    return AIMessage(content=f"[{name}] {text}")


def _bump_source(stats: dict, url: str, official) -> None:
    """Count a source as official / job_board / third_party (retrieval transparency)."""
    host = host_of(url)
    key = "official" if is_official(host, official) else ("job_board" if is_job_board(host) else "third_party")
    src = stats.setdefault("sources", {})
    src[key] = src.get(key, 0) + 1


# --------------------------------------------------------------------------- #
# 1) company_research_agent — resolve domain + overview + products (targeted)
# --------------------------------------------------------------------------- #
# STRONG product-enumeration phrases only (bare "service"/"software" also appear in
# overview sentences, which would leave the overview empty).
_PRODUCT_HINTS = (
    "products include", "product include", "services include", "offers ", " offer ",
    "we offer", "offering", "our products", "product line", "sells ", "provides a ",
)


def _categorize_company_claim(statement: str) -> str:
    low = statement.lower()
    return "product" if any(h in low for h in _PRODUCT_HINTS) else "overview"


def company_research_agent(state: DiligenceState, toolbox: Toolbox, settings: Settings) -> dict:
    """Resolve the official domain, then do targeted overview + product retrieval."""
    company = state["company_name"]
    location = state.get("location", "egypt")
    loc = _loc_terms(location)
    llm = build_llm("worker", settings)
    stats: dict = {"pages_fetched": 0, "sources": {}, "dropped_off_location": 0}

    # 1) Resolve the official domain from a broad search (drives all evidence levels).
    overview_results = _search(toolbox, f"{company} company overview about")
    official = resolve_official_domain(company, overview_results)

    # 2) Targeted, location- + site-scoped product retrieval. The company name and
    # the location bias results toward the right entity (e.g. the Egypt center).
    product_query = f"{company} {loc} site:{official} products services solutions" if official else f"{company} {loc} products services solutions"
    product_results = _search(toolbox, product_query)

    # 3) Fetch the official about + product PAGES (deeper evidence), then snippets —
    # keeping only location-relevant sources (drop other-country pages).
    def _relevant(r: dict) -> bool:
        keep = location_relevant(r.get("url", ""), r.get("snippet", ""), location)
        if not keep:
            stats["dropped_off_location"] += 1
        return keep

    evidence: list[dict] = []
    fetched = 0
    for r in overview_results[:2] + product_results[:2]:
        url = r.get("url", "")
        if not _relevant(r):
            continue
        if official and is_official(host_of(url), official) and fetched < 4:
            page = _tool(toolbox, "fetch_page", url=url)
            if page and page.get("ok") and page.get("text"):
                evidence.append({"i": len(evidence), "snippet": page["text"][:600], "source": url})
                fetched += 1
    for r in overview_results + product_results:
        if r.get("snippet") and location_relevant(r.get("url", ""), r["snippet"], location):
            evidence.append({"i": len(evidence), "snippet": r["snippet"], "source": r.get("url", "")})
    stats["pages_fetched"] = fetched

    instruction = (
        "You are the company-research specialist. Capture what the company does, "
        "its domain/business model, and its concrete products or services."
    )
    stubs = _extract_claims(llm, instruction, company, evidence)

    claims: list[Claim] = []
    seen: set[str] = set()
    for stub in stubs:
        ev = _evidence_for(stub, evidence)
        if ev is None:
            continue
        statement = str(stub.get("statement", "")).strip()
        if not statement or statement in seen:
            continue
        seen.add(statement)
        _etype, level, _note = classify_evidence(ev["source"], official)
        claims.append(Claim(
            statement=statement, source=ev["source"],
            confidence=float(stub.get("confidence", 0.7) or 0.7),
            category=_categorize_company_claim(statement), evidence_level=level,
        ))
        _bump_source(stats, ev["source"], official)

    products = [c.statement for c in claims if c.category == "product"]
    uncertainties: list[str] = []
    if not products:
        uncertainties.append(
            f"No products or services were found on the official domain "
            f"({official or 'domain unresolved'})."
        )

    return {
        "claims": claims,
        "official_domain": official,
        "uncertainties": uncertainties,
        "retrieval_stats": stats,
        "visited": ["company_research_agent"],
        "messages": [_status(
            "company_research_agent",
            f"domain={official or 'unresolved'}; {len(claims)} claim(s); {len(products)} product(s); fetched {fetched} page(s).",
        )],
    }


# --------------------------------------------------------------------------- #
# 2) technical_signals_agent — targeted tech/AI + Egypt jobs (procedural)
# --------------------------------------------------------------------------- #
def technical_signals_agent(state: DiligenceState, toolbox: Toolbox, settings: Settings) -> dict:
    """Deeper, domain-graded tech/AI detection + Egypt-scoped hiring signals.

    LLM-free: signals come from gathered text graded by source domain, jobs from a
    dedicated tool. Inferred can never become confirmed; noise is filtered.
    """
    company = state["company_name"]
    official = state.get("official_domain")
    location = state.get("location", "egypt")
    loc = _loc_terms(location)
    track = state.get("target_track") or ""
    tokens = company_tokens(company)
    stats: dict = {"pages_fetched": 0, "sources": {}, "jobs_by_source": {}, "dropped_off_location": 0}

    if not official:  # fallback if company_research couldn't resolve it
        official = resolve_official_domain(company, _search(toolbox, f"{company} official site"))

    # Keep the company name + location in the site-scoped queries so the search
    # identifies the company AND biases toward the location (e.g. the Egypt entity).
    tech_results = _search(toolbox, f"{company} {loc} site:{official} engineering blog technology" if official else f"{company} {loc} technology stack frameworks tools")
    ai_results = _search(toolbox, f"{company} {loc} site:{official} AI machine learning data" if official else f"{company} {loc} artificial intelligence machine learning data")
    eng_results = _search(toolbox, f"{company} github open source repositories")

    # Drop other-country pages before detection (keep Egypt/ETIC/neutral).
    def _loc_ok(r: dict) -> bool:
        keep = location_relevant(r.get("url", ""), r.get("snippet", ""), location)
        if not keep:
            stats["dropped_off_location"] += 1
        return keep

    tech_results = [r for r in tech_results if _loc_ok(r)]
    ai_results = [r for r in ai_results if _loc_ok(r)]
    eng_results = [r for r in eng_results if location_relevant(r.get("url", ""), r.get("snippet", ""), location)]

    # Fetch top OFFICIAL tech/AI pages for deeper detection, then add all snippets.
    ev_hits: list[dict] = []
    fetched = 0
    for r in tech_results[:3] + ai_results[:3]:
        url = r.get("url", "")
        if official and is_official(host_of(url), official) and fetched < 4:
            page = _tool(toolbox, "fetch_page", url=url)
            if page and page.get("ok") and page.get("text"):
                ev_hits.append({"snippet": page["text"], "url": url})
                fetched += 1
    for r in tech_results + ai_results + eng_results:
        if r.get("snippet"):
            ev_hits.append({"snippet": r["snippet"], "url": r.get("url", "")})
    stats["pages_fetched"] = fetched

    signals = detect_tech(ev_hits, official)
    for s in signals:
        _bump_source(stats, s.source, official)

    # Egypt-scoped hiring signals. Only TECHNICAL titles become tracks.
    jobs = (_tool(toolbox, "find_jobs", company=company, location=location, track=track) or {}).get("results", [])
    hiring: dict[str, HiringTrackSignal] = {}
    for j in jobs:
        src = j.get("source", "other")
        stats["jobs_by_source"][src] = stats["jobs_by_source"].get(src, 0) + 1
        tr = technical_track(j.get("title", ""))
        if tr and tr not in hiring:
            hiring[tr] = HiringTrackSignal(
                track=tr, evidence_type="job_posting", source=j.get("url", ""),
                confidence=0.5, notes=f"{src}: {j.get('title', '')}",
            )
    hiring_list = list(hiring.values())

    # Engineering / OSS claims, with the noise filter (§ relevance).
    eng_claims: list[Claim] = []
    oss_claims: list[Claim] = []
    for r in eng_results:
        url, snip, title = r.get("url", ""), r.get("snippet", ""), r.get("title", "")
        if not snip or not keep_signal(url, title, snip, tokens, official):
            continue
        _etype, level, _note = classify_evidence(url, official)
        is_github = "github.com" in host_of(url)
        (oss_claims if is_github else eng_claims).append(Claim(
            statement=snip, source=url, confidence=0.7,
            category=("open_source" if is_github else "engineering"), evidence_level=level,
        ))
        _bump_source(stats, url, official)

    # Honest, Egypt-scoped coverage notes for each empty section.
    ai_sigs = [s for s in signals if s.category in _AI_CATEGORIES]
    tech_sigs = [s for s in signals if s.category not in _AI_CATEGORIES]
    unc: list[str] = []
    if not tech_sigs:
        unc.append(f"No tech-stack signals were found on the official domain or job boards ({location}-scoped search).")
    if not ai_sigs:
        unc.append(f"No AI or data signals were found on the official domain or job boards ({location}-scoped search).")
    if not jobs:
        unc.append(f"No public hiring-track signals were found ({location}-scoped search across Glassdoor/LinkedIn/Wuzzuf/Bayt).")
    elif not hiring_list:
        unc.append(f"Job postings were found but none mapped to a technical hiring track ({location}-scoped).")
    if not eng_claims and not oss_claims:
        unc.append(f"No engineering-blog or open-source signals were found tied to {company}.")

    return {
        "technical_signals": signals,
        "hiring_tracks": hiring_list,
        "claims": eng_claims + oss_claims,
        "uncertainties": unc,
        "retrieval_stats": stats,
        "visited": ["technical_signals_agent"],
        "messages": [_status(
            "technical_signals_agent",
            f"{len(signals)} tech ({len(ai_sigs)} AI/data), {len(hiring_list)} track(s) from {len(jobs)} job(s), "
            f"{len(eng_claims)} eng + {len(oss_claims)} OSS, fetched {fetched} page(s).",
        )],
    }


# --------------------------------------------------------------------------- #
# 3) writer_agent — assemble the brief (deterministic) + retrieval stats
# --------------------------------------------------------------------------- #
def writer_agent(state: DiligenceState, settings: Settings) -> dict:
    """Assemble the CompanyTechIntelligenceBrief from collected evidence."""
    company = state["company_name"]
    claims = state.get("claims", [])
    signals = state.get("technical_signals", [])
    hiring = state.get("hiring_tracks", [])
    uncertainties = list(dict.fromkeys(state.get("uncertainties", [])))
    target_track = state.get("target_track")

    overview_claims = [c for c in claims if c.category == "overview"]
    product_claims = [c for c in claims if c.category == "product"]
    eng_claims = [c for c in claims if c.category == "engineering"]
    oss_claims = [c for c in claims if c.category == "open_source"]

    ai_signals = [s for s in signals if s.category in _AI_CATEGORIES]
    tech_signals = [s for s in signals if s.category not in _AI_CATEGORIES]

    overview = " ".join(c.statement for c in overview_claims).strip()
    products = list(dict.fromkeys(c.statement for c in product_claims))

    needs_review = (not overview_claims) or (not signals and not hiring)

    role_lens = _build_role_lens(target_track, tech_signals, ai_signals, company) if target_track else None

    confidences = [c.confidence for c in claims] + [s.confidence for s in signals]
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    if needs_review:
        confidence = min(confidence, 0.4)

    sources = list(dict.fromkeys(
        [c.source for c in claims if c.source]
        + [s.source for s in signals if s.source]
        + [h.source for h in hiring if h.source]
    ))

    # Retrieval transparency: merge the accumulated counters + surface Egypt scope
    # and which search provider was used (canned in stub mode).
    stats = dict(state.get("retrieval_stats") or {})
    stats["location"] = state.get("location", "")
    stats["official_domain"] = state.get("official_domain")
    if settings.tool_mode == "stub":
        stats["search_provider"] = "stub (canned)"
    else:
        from .search import active_provider

        stats["search_provider"] = active_provider()

    brief = CompanyTechIntelligenceBrief(
        company_name=company,
        company_overview=overview or "No grounded overview information was collected for this company.",
        products_and_services=products,
        technical_signals=tech_signals,
        ai_and_data_signals=ai_signals,
        hiring_tracks=hiring,
        engineering_signals=eng_claims,
        open_source_signals=oss_claims,
        uncertainties=uncertainties,
        role_lens=role_lens,
        sources=sources,
        confidence=confidence,
        needs_human_review=needs_review,
        retrieval_stats=stats,
    )
    return {
        "brief": brief,
        "messages": [_status("writer_agent", f"brief assembled; role_lens={'yes' if role_lens else 'no'}; review={needs_review}.")],
    }


# --------------------------------------------------------------------------- #
# Role lens — OPTIONAL interview-prep analysis (not a sourced factual claim)
# --------------------------------------------------------------------------- #
# Small in-code track profiles. # V1.5: move these to profiles/*.yaml.
_GENERIC_PROFILE = {
    "skills": [],
    "questions": [
        "Walk me through a project most relevant to this company's domain.",
        "How do you keep technical work grounded and verifiable?",
    ],
    "talking_points": ["Tie your portfolio to the company's public products and technical direction."],
    "ask_company": ["What does the team's current tech stack and roadmap look like?"],
}
TRACK_PROFILES: dict[str, dict] = {
    "ai_engineer": {
        "skills": ["LLMs", "RAG", "prompt engineering", "vector databases", "Python", "evaluation/guardrails"],
        "questions": [
            "How would you design a RAG pipeline and evaluate its groundedness?",
            "How do you prevent hallucination in a production LLM feature?",
            "When would you fine-tune vs. use retrieval?",
        ],
        "talking_points": ["Show a grounded, evaluated LLM/RAG project with sources and metrics."],
        "ask_company": ["How do you evaluate and monitor your AI features in production?"],
    },
    "data_engineer": {
        "skills": ["SQL", "Python", "ETL/ELT", "Airflow", "data warehousing", "Spark/Kafka"],
        "questions": [
            "How would you design an idempotent, backfillable data pipeline?",
            "How do you handle data quality and schema drift?",
        ],
        "talking_points": ["Highlight a pipeline you built end-to-end with reliability/quality checks."],
        "ask_company": ["What does your data platform and orchestration stack look like?"],
    },
    "backend_engineer": {
        "skills": ["API design", "databases", "microservices", "testing", "one of Java/Python/Go"],
        "questions": [
            "How would you design a scalable, well-tested REST/gRPC service?",
            "How do you reason about consistency, caching, and failure modes?",
        ],
        "talking_points": ["Show a service you designed with clear API contracts and tests."],
        "ask_company": ["How are services structured, deployed, and observed here?"],
    },
    "odoo_functional": {
        "skills": ["Odoo modules", "business-process mapping", "PostgreSQL", "workflow automation", "ERP integration"],
        "questions": [
            "How do you translate a business process into Odoo configuration vs. custom code?",
            "How would you approach an Odoo-to-SAP integration?",
        ],
        "talking_points": ["Show an ERP/Odoo implementation where you mapped real processes to the system."],
        "ask_company": ["Which Odoo modules and integrations are most critical to your clients?"],
    },
    "sap_consultant": {
        "skills": ["SAP modules", "integration", "business-process analysis", "data migration"],
        "questions": [
            "How do you scope an SAP integration with an existing ERP landscape?",
            "How do you de-risk a data migration?",
        ],
        "talking_points": ["Show an enterprise integration/migration you helped deliver."],
        "ask_company": ["What does your SAP landscape and integration roadmap look like?"],
    },
}


def _normalize_track(track: str) -> str:
    return re.sub(r"[\s/\-]+", "_", track.strip().lower())


def _build_role_lens(track, tech_signals, ai_signals, company) -> RoleLens:
    """Build the optional interview-prep lens (skills + questions guaranteed non-empty)."""
    profile = TRACK_PROFILES.get(_normalize_track(track), _GENERIC_PROFILE)
    company_techs = [s.technology for s in (ai_signals + tech_signals)]

    skills = list(dict.fromkeys(profile["skills"] + company_techs))
    if not skills:
        skills = ["Review the company's public products and technical direction"]

    questions = list(profile["questions"])
    if company_techs:
        questions.append(f"How does {company} use {company_techs[0]} in production?")

    talking = list(profile["talking_points"])
    if company_techs:
        talking.append(f"Relate your work to {company}'s stack: {', '.join(company_techs[:4])}.")

    techs_phrase = ", ".join(company_techs[:5]) if company_techs else "limited public technical signals"
    fit = (
        f"Based on public signals, {company} shows {techs_phrase}. For the "
        f"'{track}' track, emphasise the overlapping skills above."
    )
    return RoleLens(
        target_track=track, fit_summary=fit, skills_to_prepare=skills,
        portfolio_talking_points=talking, interview_questions_to_prepare=questions,
        questions_to_ask_company=list(profile["ask_company"]), caveat=ROLE_LENS_CAVEAT,
    )
