"""Offline evaluation for the DiligenceDesk technical-intelligence system.

Scores the real graph (stub mode) over fixed scenarios on a panel of properties:

    routing_correctness    supervisor routed to both specialists + writer?
    groundedness           every cited source traces to a tool result, AND every
                           CONFIRMED signal's host equals the company's official
                           domain (a third-party mention is never confirmed), AND
                           a known third-party tech stays 'inferred'.
    tech_signal_recall     expected technologies detected?
    ai_data_signal_recall  AI/data techs found when present; NOT invented when absent.
    hiring_track_detection expected (technical) hiring tracks surfaced?
    uncertainty_honesty    absent signals are stated honestly (Egypt-scoped), and no
                           false gap is claimed.
    products_extracted     products_and_services filled for companies with a products page.
    role_lens_completeness no track -> null; track -> populated skills + questions.
    report_validity        the brief satisfies its Pydantic schema.

Deterministic + fully offline. LangSmith optional.

Run it:  python eval/evaluate.py   [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["TOOL_MODE"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
for _role in ("SUPERVISOR", "WORKER", "WRITER"):
    os.environ[f"{_role}_PROVIDER"] = "stub"

from diligencedesk.graph import build_graph  # noqa: E402
from diligencedesk.retrieval import host_of, is_official  # noqa: E402
from diligencedesk.state import initial_state  # noqa: E402
from diligencedesk.tools import stub_tools  # noqa: E402

RESEARCH_SPECIALISTS = {"company_research_agent", "technical_signals_agent"}


def run_company(app, company: str, target_track=None):
    thread = f"eval-{company}-{target_track or 'none'}"
    config = {"configurable": {"thread_id": thread}}
    app.invoke(initial_state(company, target_track, "egypt"), config=config)
    return app.get_state(config).values


def score(sc: dict, final: dict, allowed: set[str]) -> dict:
    brief = final["brief"]
    claims = final.get("claims", [])
    signals = final.get("technical_signals", [])
    hiring = final.get("hiring_tracks", [])
    visited = set(final.get("visited", []))
    expected = sc["expected"]
    official = expected["official_domain"]

    all_signals = list(brief.technical_signals) + list(brief.ai_and_data_signals)
    techs = {s.technology for s in all_signals}
    ai_techs = {s.technology for s in brief.ai_and_data_signals}
    tracks = {h.track for h in brief.hiring_tracks}

    # 1) routing
    routing_ok = visited == RESEARCH_SPECIALISTS and brief is not None

    # 2) groundedness = (a) all sources real, (b) every CONFIRMED signal/claim is on
    # the official domain, (c) the known third-party tech stays inferred.
    cited = [c.source for c in claims] + [s.source for s in signals] + [h.source for h in hiring]
    cited = [s for s in cited if s]
    sources_valid = all(s in allowed for s in cited)

    confirmed_items = [s for s in all_signals if s.evidence_level == "confirmed"]
    confirmed_items += [c for c in (brief.engineering_signals + brief.open_source_signals) if c.evidence_level == "confirmed"]
    domain_confirmed_ok = all(is_official(host_of(x.source), official) for x in confirmed_items)

    inferred_tech = expected.get("inferred_not_confirmed")
    inferred_ok = True
    if inferred_tech:
        match = [s for s in all_signals if s.technology == inferred_tech]
        inferred_ok = bool(match) and all(s.evidence_level == "inferred" for s in match)

    groundedness_ok = sources_valid and domain_confirmed_ok and inferred_ok

    # 3) tech recall / 4) ai recall
    tech_ok = all(t in techs for t in expected["tech_signals"])
    if expected["ai_signals"]:
        ai_ok = all(t in ai_techs for t in expected["ai_signals"])
    else:
        ai_ok = len(ai_techs) == 0

    # 5) hiring-track detection
    hiring_ok = all(t in tracks for t in expected["hiring_tracks"])

    # 6) uncertainty honesty
    unc_text = " | ".join(brief.uncertainties)
    required_ok = all(sub in unc_text for sub in expected["uncertainty_contains"])
    no_false_ai_gap = not (ai_techs and "No AI or data" in unc_text)
    uncertainty_ok = required_ok and no_false_ai_gap

    # 7) products extracted
    products_ok = (len(brief.products_and_services) > 0) if expected["products_expected"] else True

    # 8) role_lens (no-track run must be null)
    role_lens_none_ok = brief.role_lens is None

    try:
        type(brief).model_validate(brief.model_dump())
        report_valid = True
    except Exception:  # noqa: BLE001
        report_valid = False

    return {
        "routing_ok": routing_ok, "groundedness_ok": groundedness_ok, "tech_ok": tech_ok,
        "ai_ok": ai_ok, "hiring_ok": hiring_ok, "uncertainty_ok": uncertainty_ok,
        "products_ok": products_ok, "role_lens_none_ok": role_lens_none_ok,
        "report_valid": report_valid,
        "ungrounded": [s for s in cited if s not in allowed],
        "confirmed_offdomain": [x.source for x in confirmed_items if not is_official(host_of(x.source), official)],
    }


def score_with_track(app, sc: dict) -> bool | None:
    track = sc["expected"].get("role_lens_track")
    if not track:
        return None
    final = run_company(app, sc["company"], target_track=track)
    rl = final["brief"].role_lens
    return rl is not None and len(rl.skills_to_prepare) > 0 and len(rl.interview_questions_to_prepare) > 0


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _fmt(value):
    return "n/a" if value is None else f"{value:.0%}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DiligenceDesk (offline, stub mode).")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N scenarios.")
    args = parser.parse_args(argv)

    stub_tools.reset_corpus()  # use the built-in synthetic companies
    scenarios = json.loads((Path(__file__).resolve().parent / "scenarios.json").read_text("utf-8"))["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]

    app = build_graph()

    print("=" * 82)
    print(f"DiligenceDesk eval - {len(scenarios)} scenario(s), stub mode (deterministic)")
    print("=" * 82)

    records = []
    with_track = []
    for sc in scenarios:
        fx = stub_tools.fixture_for(sc["company"])
        allowed = stub_tools.all_source_urls(fx) if fx else set()
        final = run_company(app, sc["company"])
        m = score(sc, final, allowed)
        wt = score_with_track(app, sc)
        if wt is not None:
            with_track.append(wt)
        records.append({"id": sc["id"], "company": sc["company"], "metrics": m})
        flags = "".join("Y" if m[k] else "N" for k in
                        ("routing_ok", "groundedness_ok", "tech_ok", "ai_ok", "hiring_ok", "uncertainty_ok", "products_ok", "role_lens_none_ok", "report_valid"))
        extra = "" if wt is None else f"  (with-track lens: {'Y' if wt else 'N'})"
        print(f"[{sc['id']:<22}] route/ground/tech/ai/hiring/uncert/prod/lens/valid = {flags}{extra}")
        if m["confirmed_offdomain"]:
            print(f"    !! CONFIRMED signal off the official domain: {m['confirmed_offdomain']}")
        if m["ungrounded"]:
            print(f"    !! ungrounded sources: {m['ungrounded']}")

    def col(key):
        return _mean([r["metrics"][key] for r in records])

    role_lens_vals = [r["metrics"]["role_lens_none_ok"] for r in records] + with_track
    table = {
        "routing_correctness": col("routing_ok"),
        "groundedness": col("groundedness_ok"),
        "tech_signal_recall": col("tech_ok"),
        "ai_data_signal_recall": col("ai_ok"),
        "hiring_track_detection": col("hiring_ok"),
        "uncertainty_honesty": col("uncertainty_ok"),
        "products_extracted": col("products_ok"),
        "role_lens_completeness": _mean(role_lens_vals),
        "report_validity": col("report_valid"),
    }
    print("-" * 82)
    print("METRIC                          VALUE")
    for name, value in table.items():
        print(f"  {name:<30}{_fmt(value)}")
    print("-" * 82)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"eval_{stamp}.json"
    out_path.write_text(json.dumps({"run": {"timestamp": stamp, "mode": "stub"}, "metrics": table, "cases": records}, indent=2), encoding="utf-8")
    print(f"Saved full record -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
