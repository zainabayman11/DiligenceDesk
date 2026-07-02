"""Command-line entry point for DiligenceDesk.

Keyless, fully-offline demo (stub LLM + stub tools, deterministic):
    python run.py --mode stub --company "Acme AI Health"
    python run.py --mode stub --company "Acme AI Health" --target-track ai_engineer

Real research with zero MCP setup (real tools; needs an LLM key in .env):
    python run.py --mode local --provider groq --company "Nvidia" --target-track ai_engineer --save

The MCP learning path (fetch + filesystem via real MCP servers; needs uvx/Node):
    python run.py --mode mcp --provider groq --company "Airbnb"

The company brief is GENERAL technical intelligence. If --target-track is given,
the brief adds an OPTIONAL role-specific interview-preparation lens (a bonus
layer, not the core product).

WHY a streaming CLI: we print each routing decision and each specialist's status
as it happens, so you can SEE the supervisor pattern at work before the final
brief is rendered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make the src/ layout importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from diligencedesk.config import get_settings  # noqa: E402
from diligencedesk.graph import build_graph  # noqa: E402
from diligencedesk.schemas import CompanyTechIntelligenceBrief  # noqa: E402
from diligencedesk.state import initial_state  # noqa: E402
from diligencedesk.tools import stub_tools  # noqa: E402
from diligencedesk.tools.provider import get_tools  # noqa: E402


def render_brief(brief: CompanyTechIntelligenceBrief) -> str:
    """Render the structured brief as readable Markdown.

    The brief is honest by construction: each technical signal shows its evidence
    level (confirmed vs inferred), uncertainties list what was NOT found, and the
    optional role lens is clearly marked as interpretation, not sourced fact.
    """
    lines = [
        f"# Company Technical-Intelligence Brief: {brief.company_name}",
        "",
        "> Factual technical intelligence from public sources. The role lens (if "
        "present) is interpretation for interview prep, not a sourced company claim.",
        "",
        f"**Overall confidence:** {brief.confidence:.0%}"
        + ("  [!] NEEDS HUMAN REVIEW" if brief.needs_human_review else ""),
        "",
        "## Overview",
        brief.company_overview,
    ]

    lines += ["", "## Products & services"]
    lines += [f"- {p}" for p in brief.products_and_services] or ["- (none found)"]

    lines += ["", "## Technical signals"]
    if brief.technical_signals:
        lines += [f"- **[{s.evidence_level}]** {s.technology} ({s.category})  ({s.source})"
                  for s in brief.technical_signals]
    else:
        lines.append("- No clear public tech-stack signals were found.")

    lines += ["", "## AI & data signals"]
    if brief.ai_and_data_signals:
        lines += [f"- **[{s.evidence_level}]** {s.technology} ({s.category})  ({s.source})"
                  for s in brief.ai_and_data_signals]
    else:
        lines.append("- No public AI or data signals were found.")

    lines += ["", "## Hiring tracks"]
    lines += [f"- {h.track} (via {h.evidence_type})  ({h.source})" for h in brief.hiring_tracks] \
        or ["- No public hiring-track signals were found."]

    if brief.engineering_signals:
        lines += ["", "## Engineering signals"]
        lines += [f"- **[{c.evidence_level}]** {c.statement}  ({c.source})" for c in brief.engineering_signals]
    if brief.open_source_signals:
        lines += ["", "## Open-source signals"]
        lines += [f"- **[{c.evidence_level}]** {c.statement}  ({c.source})" for c in brief.open_source_signals]

    lines += ["", "## Uncertainties / evidence gaps"]
    lines += [f"- {u}" for u in brief.uncertainties] or ["- (none noted)"]

    if brief.role_lens:
        rl = brief.role_lens
        lines += [
            "", f"## Role lens: {rl.target_track}  (interview preparation)",
            f"_{rl.caveat}_", "",
            f"**Fit summary:** {rl.fit_summary}",
            "", "**Skills to prepare:**", *[f"- {s}" for s in rl.skills_to_prepare],
            "", "**Interview questions to prepare:**", *[f"- {q}" for q in rl.interview_questions_to_prepare],
        ]
        if rl.portfolio_talking_points:
            lines += ["", "**Portfolio talking points:**", *[f"- {t}" for t in rl.portfolio_talking_points]]
        if rl.questions_to_ask_company:
            lines += ["", "**Questions to ask the company:**", *[f"- {q}" for q in rl.questions_to_ask_company]]

    lines += ["", "## Sources"]
    lines += [f"- {s}" for s in brief.sources] or ["- (none)"]

    stats = brief.retrieval_stats or {}
    lines += [
        "", "## Retrieval",
        f"- **Search provider:** {stats.get('search_provider') or 'n/a'}",
        f"- **Official domain:** {stats.get('official_domain') or '(unresolved)'}",
        f"- **Job location scope:** {stats.get('location') or 'n/a'}",
        f"- **Pages fetched:** {stats.get('pages_fetched', 0)}",
        f"- **Sources by type:** {stats.get('sources', {})}",
        f"- **Jobs per source:** {stats.get('jobs_by_source', {})}",
    ]
    return "\n".join(lines)


def _slug(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in name.strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "company"


def save_trace(company, target_track, settings, steps, brief) -> Path:
    """Persist the full run trace (routing path + agent status + brief) to a file.

    ALWAYS runs, and NEVER overwrites: each run writes its own timestamped
    outputs/trace_<company>_<YYYYmmdd_HHMMSS>.json. This is the auditable record of
    what the supervisor routed, what each specialist reported, and the final brief.
    """
    outputs = Path(__file__).resolve().parent / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = outputs / f"trace_{_slug(company)}_{stamp}.json"
    payload = {
        "company": company,
        "target_track": target_track,
        "tool_mode": settings.tool_mode,
        "supervisor": f"{settings.role('supervisor').provider}/{settings.role('supervisor').model}",
        "worker": f"{settings.role('worker').provider}/{settings.role('worker').model}",
        "timestamp": stamp,
        "steps": steps,
        "brief": brief.model_dump() if brief else None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(company: str, target_track: str | None = None, location: str = "egypt", thread: str = "cli") -> CompanyTechIntelligenceBrief:
    """Run the team for one company, streaming the trace and saving it to disk.

    # V1.5: a FastAPI `POST /company-tech-brief {company, target_track, location}`
    # endpoint would wrap exactly this function, returning brief.model_dump() as JSON.
    """
    settings = get_settings()
    app = build_graph(settings=settings)

    print("=" * 72)
    print("DiligenceDesk - multi-agent company technical-intelligence (supervisor pattern)")
    print(f"  company   : {company}  | job location: {location}"
          + (f"  | target track: {target_track}" if target_track else ""))
    print(f"  tool mode : {settings.tool_mode} | supervisor: {settings.role('supervisor').provider}"
          f"/{settings.role('supervisor').model} | worker: {settings.role('worker').provider}"
          f"/{settings.role('worker').model}")
    print("=" * 72)

    config = {"configurable": {"thread_id": thread}}
    steps: list[dict] = []
    for chunk in app.stream(initial_state(company, target_track, location), config=config, stream_mode="updates"):
        for node, update in chunk.items():
            step = {"node": node}
            if node == "supervisor":
                step["route"] = (update or {}).get("route")
                print(f"  [supervisor] -> {step['route']}")
            statuses = []
            for msg in (update or {}).get("messages", []):
                if isinstance(msg, AIMessage):
                    print(f"  {msg.content}")
                    statuses.append(str(msg.content))
            if statuses:
                step["status"] = statuses
            steps.append(step)

    brief = app.get_state(config).values["brief"]

    # Retrieval transparency: show the search provider + that retrieval went deep.
    stats = brief.retrieval_stats or {}
    print(
        f"\n[retrieval] search_provider={stats.get('search_provider')} | domain={stats.get('official_domain')} | "
        f"location={stats.get('location')} | pages_fetched={stats.get('pages_fetched', 0)} | "
        f"sources={stats.get('sources', {})} | jobs_by_source={stats.get('jobs_by_source', {})}"
    )

    trace_path = save_trace(company, target_track, settings, steps, brief)
    print(f"[trace saved] {trace_path}")
    return brief


def _force_all_roles(provider: str) -> None:
    """Force every role (supervisor/worker/writer) onto one provider.

    Used when the CLI picks a single provider (or the offline stub), so it wins
    over any per-role SUPERVISOR_PROVIDER/WORKER_PROVIDER set in .env. We also
    CLEAR the per-role model overrides, so the provider's own default model is
    used — otherwise a .env `SUPERVISOR_MODEL=gemini-...` would be sent to Groq.
    """
    for role in ("SUPERVISOR", "WORKER", "WRITER"):
        os.environ[f"{role}_PROVIDER"] = provider
        os.environ.pop(f"{role}_MODEL", None)
    os.environ["LLM_PROVIDER"] = provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiligenceDesk - company technical-intelligence assistant.")
    parser.add_argument("--company", default="Acme AI Health", help="Company to research (required in practice).")
    parser.add_argument(
        "--target-track", default=None,
        help="Optional role track for an interview-prep lens, e.g. ai_engineer, "
        "backend_engineer, data_engineer, odoo_functional, sap_consultant.",
    )
    parser.add_argument(
        "--location", default="egypt",
        help="Location scope for the job search (default: egypt).",
    )
    parser.add_argument("--mode", choices=["stub", "local", "mcp"], help="Tool layer mode (overrides TOOL_MODE).")
    parser.add_argument(
        "--provider", choices=["groq", "gemini", "ollama", "stub"],
        help="LLM provider for every role (overrides LLM_PROVIDER).",
    )
    parser.add_argument("--thread", default="cli", help="Thread id for the checkpointer.")
    parser.add_argument("--save", action="store_true", help="Also save the brief to outputs/ via the save_brief tool.")
    parser.add_argument(
        "--list-companies", action="store_true",
        help="List the synthetic companies available in stub mode and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Real web data contains characters a legacy Windows console codepage (e.g.
    # cp1256 / cp1252) can't encode, which would crash the final print. Force UTF-8
    # with safe replacement so the CLI never dies on an odd character.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - older Python / non-reconfigurable stream
            pass

    args = parse_args(argv)

    if args.list_companies:
        print("Stub-mode companies:")
        for name in stub_tools.builtin_company_names():
            print(f"  - {name}")
        return 0

    # CLI flags win, applied via the environment so there is ONE resolution path.
    # DILIGENCE_LOCATION lets the search provider region-scope even the plain
    # web_search tool (which takes no location argument).
    os.environ["DILIGENCE_LOCATION"] = args.location
    if args.mode:
        os.environ["TOOL_MODE"] = args.mode
    effective_mode = os.environ.get("TOOL_MODE", "stub")
    if args.provider:
        # An explicit provider forces EVERY role to it (overrides any .env per-role).
        _force_all_roles(args.provider)
    elif effective_mode == "stub":
        # stub tools imply a fully-offline stub LLM: a keyless demo needs no keys,
        # regardless of the Gemini/Groq per-role setup in .env for real runs.
        _force_all_roles("stub")
    # else (local/mcp with no --provider): use the per-role providers from .env,
    # e.g. SUPERVISOR_PROVIDER=gemini + WORKER_PROVIDER=groq.

    try:
        brief = run(args.company, target_track=args.target_track, location=args.location, thread=args.thread)
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    print("\n" + render_brief(brief) + "\n")

    if args.save:
        settings = get_settings()
        toolbox = get_tools(settings.tool_mode, settings)
        result = toolbox.get("save_brief").invoke(
            {"content": render_brief(brief), "filename": brief.company_name}
        )
        print(f"[save_brief] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
