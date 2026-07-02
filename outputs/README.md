# Example outputs

Sample runs kept for reference. Each run produces two files:

- `trace_<company>_<timestamp>.json` — the full run trace: the supervisor's routing
  path, each specialist's status line, the assembled `CompanyTechIntelligenceBrief`,
  and `retrieval_stats` (search provider, resolved official domain, pages fetched,
  sources by type, jobs per board).
- `<company>_<timestamp>.md` — the rendered brief (only present when a run used `--save`).

## What's here

| Company | Mode | Notes |
| --- | --- | --- |
| **Acme AI Health** | stub | AI/data-heavy: confirmed Python/FastAPI/RAG, `Kubernetes` stays *inferred* (third-party), Egypt hiring tracks. |
| **Globex Cloud Systems** | stub | Backend/cloud; no public AI → honest "no AI signals" note. |
| **Initech ERP Services** | stub | Odoo/SAP/ERP. |
| **NoSignal Consulting** | stub | Limited info → every empty section gets an honest coverage note. |
| **PwC ETIC** | local (real web) | Real run: domain resolved to `pwc.com`, Egypt-scoped jobs across 4 boards, US pages filtered out, US job postings correctly graded *inferred*. |

> Regenerate any of these with, e.g.:
> `python run.py --mode stub --company "Acme AI Health" --location egypt --target-track ai_engineer`

These are deterministic in `stub` mode; `local`/`mcp` runs vary with live web results.
