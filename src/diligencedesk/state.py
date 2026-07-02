"""The shared graph state for the technical-intelligence team.

WHY this shape (MessagesState-style + typed evidence stores):
`messages` stay lean — only compact status lines, NOT each specialist's raw tool
traffic. The real work product lives in typed, append-only stores: `claims`,
`technical_signals`, `hiring_tracks`, `uncertainties`. The writer builds the brief
from those stores, never from free-form chatter, so every factual line traces back
to a sourced object.

Reducers:
- `messages` -> add_messages (append + de-dupe).
- `claims`, `technical_signals`, `hiring_tracks`, `uncertainties`, `visited` ->
  operator.add (each specialist APPENDS its contribution).
- `retrieval_stats` -> _merge_stats (both agents contribute counters; we sum them).
- everything else (brief, turns, route, company_name, target_track, location,
  official_domain) -> default "last write wins".
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from .schemas import Claim, CompanyTechIntelligenceBrief, HiringTrackSignal, TechnicalSignal


def _merge_stats(old: dict | None, new: dict | None) -> dict:
    """Reducer that deep-merges retrieval counters, summing numeric leaves.

    So the two specialists' {pages_fetched, sources:{...}, jobs_by_source:{...}}
    add up into one transparent total, instead of overwriting each other.
    """
    if not old:
        return new or {}
    if not new:
        return old
    out = dict(old)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            sub = dict(out[key])
            for k, v in value.items():
                if isinstance(v, (int, float)) and isinstance(sub.get(k), (int, float)):
                    sub[k] = sub[k] + v
                else:
                    sub[k] = v
            out[key] = sub
        elif isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + value
        else:
            out[key] = value
    return out


class DiligenceState(TypedDict, total=False):
    """State flowing through the supervisor + specialist graph."""

    # The company under review + optional inputs (role lens, job-search location).
    company_name: str
    target_track: Optional[str]
    location: str
    # The company's resolved official domain (set by company_research_agent). This
    # is what makes evidence levels domain-derived instead of assumed.
    official_domain: Optional[str]
    # Lean conversation: compact status lines from the supervisor/specialists.
    messages: Annotated[list, add_messages]
    # Append-only structured evidence stores (the grounding backbone).
    claims: Annotated[list[Claim], operator.add]
    technical_signals: Annotated[list[TechnicalSignal], operator.add]
    hiring_tracks: Annotated[list[HiringTrackSignal], operator.add]
    uncertainties: Annotated[list[str], operator.add]
    # Retrieval transparency counters (merged across specialists).
    retrieval_stats: Annotated[dict, _merge_stats]
    # Which specialists have contributed (the supervisor reads this to route).
    visited: Annotated[list[str], operator.add]
    # Supervisor bookkeeping: turn counter (cost guard) + last routing decision.
    turns: int
    route: str
    # The final deliverable (set by the writer).
    brief: Optional[CompanyTechIntelligenceBrief]


def initial_state(company_name: str, target_track: Optional[str] = None, location: str = "egypt") -> DiligenceState:
    """Build a fresh state for a run.

    Append-only lists are initialised explicitly so their reducers always have a
    base to append onto. `location` defaults to Egypt (the job-search scope).
    """
    return {
        "company_name": company_name,
        "target_track": target_track,
        "location": location,
        "official_domain": None,
        "messages": [],
        "claims": [],
        "technical_signals": [],
        "hiring_tracks": [],
        "uncertainties": [],
        "retrieval_stats": {},
        "turns": 0,
        "route": "",
        "brief": None,
    }
