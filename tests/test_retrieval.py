"""Retrieval-logic tests — the correctness rules, in isolation (no graph/network).

These lock in the two most important guarantees: evidence level is DOMAIN-derived
(a third-party mention is never confirmed), and noise/non-technical roles are filtered.
"""

from __future__ import annotations

from diligencedesk.retrieval import (
    classify_evidence,
    company_tokens,
    detect_tech,
    keep_signal,
    location_relevant,
    resolve_official_domain,
    technical_track,
)


def test_location_relevance_keeps_egypt_drops_other_countries():
    # A US / Czech page (other country) is dropped under the Egypt scope...
    assert not location_relevant("https://jobs.us.pwc.com/job/new-york/x", "ML developer", "egypt")
    assert not location_relevant("https://www.pwc.com/cz/en/services/data.html", "analytics", "egypt")
    # ...an Egypt / ETIC / Middle-East page is kept...
    assert location_relevant("https://www.pwc.com/m1/en/careers/egypt-technology-innovation-centre.html", "ETIC Egypt", "egypt")
    # ...a location-neutral company page is kept...
    assert location_relevant("https://acme-ai-health.com/engineering", "Python FastAPI", "egypt")
    # ...and a non-Egypt scope disables the filter entirely.
    assert location_relevant("https://jobs.us.pwc.com/job/x", "", "usa")


def test_official_is_confirmed_third_party_and_jobboard_are_inferred():
    official = "acme.com"
    _et, level, note = classify_evidence("https://acme.com/engineering/blog/x", official)
    assert level == "confirmed" and "engineering" in note

    _et, level, note = classify_evidence("https://www.wsj.com/articles/acme", official)
    assert level == "inferred" and "third-party" in note  # NOT hardcoded "official site"

    et, level, _note = classify_evidence("https://www.linkedin.com/jobs/view/1", official)
    assert et == "job_posting" and level == "inferred"


def test_detect_tech_never_upgrades_a_third_party_mention():
    official = "acme.com"
    hits = [
        {"snippet": "Our stack is Python and FastAPI.", "url": "https://acme.com/engineering"},
        {"snippet": "Acme reportedly uses Kubernetes and TensorFlow.", "url": "https://www.wsj.com/x"},
    ]
    levels = {s.technology: s.evidence_level for s in detect_tech(hits, official)}
    assert levels["Python"] == "confirmed"       # company's own page
    assert levels["Kubernetes"] == "inferred"    # third-party -> never confirmed
    assert levels["TensorFlow"] == "inferred"


def test_careers_url_is_inferred_even_on_the_official_domain():
    official = "pwc.com"
    # A job posting on the company's OWN careers subdomain must be inferred,
    # never confirmed (it's aspirational — "we're hiring for X", not proof).
    et, level, _ = classify_evidence(
        "https://jobs.us.pwc.com/job/new-york/ml-developer/932/95591450096", official)
    assert et == "job_posting" and level == "inferred"
    # A /careers path on the official domain -> inferred too.
    _et, level2, _ = classify_evidence("https://pwc.com/eg/careers/data-engineer", official)
    assert level2 == "inferred"
    # ...but a genuine services/tech page on the official domain stays confirmed.
    _et, level3, _ = classify_evidence("https://pwc.com/eg/services/technology", official)
    assert level3 == "confirmed"


def test_technical_track_excludes_non_technical_roles():
    assert technical_track("Machine Learning Engineer") == "AI Engineer"
    assert technical_track("DevOps Engineer") == "DevOps / Platform Engineer"
    assert technical_track("Backend Developer") is not None
    # Non-technical roles must NEVER become a technical hiring track.
    assert technical_track("HR Manager") is None
    assert technical_track("Sales Account Manager") is None
    assert technical_track("Marketing Specialist") is None


def test_keep_signal_drops_unrelated_repo_keeps_company_org():
    tokens = company_tokens("Acme AI Health")
    official = "acme-ai-health.com"
    assert keep_signal("https://github.com/acme-ai-health/nlp", "nlp", "acme tools", tokens, official)
    assert keep_signal("https://acme-ai-health.com/blog", "b", "a post", tokens, official)
    # An aggregator repo that merely LISTS the company name is dropped.
    assert not keep_signal(
        "https://github.com/someone/awesome-list", "awesome",
        "a curated list including Acme AI Health and others", tokens, official,
    )


def test_resolve_official_domain_skips_wikipedia_and_news():
    results = [
        {"url": "https://en.wikipedia.org/wiki/Acme_AI_Health"},
        {"url": "https://techcrunch.com/acme"},
        {"url": "https://acme-ai-health.com/about"},
    ]
    assert resolve_official_domain("Acme AI Health", results) == "acme-ai-health.com"
