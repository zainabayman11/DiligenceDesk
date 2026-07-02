"""A deterministic, offline fake chat model.

WHY this exists:
- The WHOLE multi-agent system can run with NO API key and NO LLM network call:
  the supervisor routes, the specialists synthesise claims, the writer assembles
  a brief — all deterministically. That is what powers `--mode stub`, the tests,
  and the offline eval.
- A predictable "model" lets tests assert on exact routing and grounding without
  mocking an HTTP endpoint or paying for tokens.

WHAT it is NOT: it does not reason. It is a tiny, transparent JSON transformer
that speaks the two protocols our nodes use:

  1. ROUTING  — the supervisor asks "which specialist next?" and passes the legal
     `options`. The stub returns the FIRST option. (The real router would weigh
     them; the stub is deterministic so routing is testable.)

  2. CLAIM EXTRACTION — a specialist passes gathered EVIDENCE (snippets + their
     real source URLs) and asks for grounded claims. The stub echoes each piece
     of evidence back as a claim, carrying the SAME source. This is grounding by
     construction: the stub literally cannot cite a source the tools didn't
     return, which is exactly the behaviour we want from the real model too.

Because the stub only transforms whatever evidence it is given, it works for ANY
company/scenario (including the eval's canned OpenAI / Nvidia / Stripe fixtures)
without hard-coding a single company fact. The fixtures carry the facts; the stub
just routes and grounds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Markers the nodes embed in their prompts so the stub knows which protocol to
# speak. The real providers ignore these (they read the surrounding instruction);
# the stub keys off them. Defined here AND referenced by the nodes via these
# constants, so the two can never drift.
ROUTE_MARKER = "TASK: ROUTE"
CLAIMS_MARKER = "TASK: EXTRACT_CLAIMS"

_OPTIONS_RE = re.compile(r"options:\s*\[([^\]]*)\]", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"<EVIDENCE>(.*?)</EVIDENCE>", re.DOTALL)


def _all_text(messages: list[BaseMessage]) -> str:
    """Flatten every message's content to one searchable string."""
    return "\n".join(str(m.content) for m in messages)


def _route_response(text: str) -> str:
    """Pick the first legal option the supervisor offered."""
    match = _OPTIONS_RE.search(text)
    options = []
    if match:
        options = [o.strip().strip("\"'") for o in match.group(1).split(",") if o.strip()]
    # No options parsed => fall back to finishing via the writer, so the graph
    # always terminates even if a prompt is malformed.
    nxt = options[0] if options else "writer_agent"
    return json.dumps({"next": nxt, "reason": "stub: first legal option"})


def _claims_response(text: str) -> str:
    """Echo each evidence item back as a grounded claim (same source)."""
    match = _EVIDENCE_RE.search(text)
    if not match:
        return "[]"
    try:
        evidence = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return "[]"

    claims = []
    for item in evidence:
        # The stub's "summary" is just the evidence snippet itself — the most
        # honest, perfectly-grounded statement possible. `i` lets the specialist
        # re-attach the real source; severity (risk only) is passed through.
        claim = {
            "i": item.get("i"),
            "statement": str(item.get("snippet", "")).strip(),
            "confidence": 0.72,
        }
        if "severity" in item:
            claim["severity"] = item["severity"]
        claims.append(claim)
    return json.dumps(claims)


class StubChatModel(BaseChatModel):
    """A fake chat model that speaks the routing + claim-extraction protocols."""

    @property
    def _llm_type(self) -> str:
        return "stub-diligencedesk"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubChatModel":
        """Accept tools but ignore them — the stub never tool-calls.

        Present only so the stub is a drop-in for a real model; our nodes drive
        it with plain .invoke() and parse the JSON content.
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = _all_text(messages)
        if ROUTE_MARKER in text:
            content = _route_response(text)
        elif CLAIMS_MARKER in text:
            content = _claims_response(text)
        else:
            # Unknown prompt: return empty JSON, never crash. Nodes parse
            # defensively, so this degrades gracefully.
            content = "[]"
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_stub_llm() -> StubChatModel:
    """Factory mirroring build_llm() so llm.py can stay symmetric."""
    return StubChatModel()
