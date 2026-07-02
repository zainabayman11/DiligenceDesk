"""DiligenceDesk — a multi-agent company technical-intelligence assistant on LangGraph.

A supervisor routes between specialist agents that research a company and gather
public TECHNICAL signals — products/services, tech stack, AI/data signals, hiring
tracks, engineering/open-source evidence, and uncertainty — then a writer assembles
a structured, source-cited technical-intelligence brief. An optional role lens adds
interview-preparation analysis when a target track is supplied.

Responsible-AI stance (baked into the design, not bolted on):
- read-only research over PUBLIC information for a user-provided company;
- the output is a factual technical summary, NOT advice;
- every factual claim in the brief is grounded in a tool result with its source;
- tech detection only surfaces technologies that appear in the gathered text (it
  never invents one), and every signal is graded confirmed / inferred / not_found
  so an inferred technology is never presented as confirmed.
"""

__version__ = "1.0.0"
