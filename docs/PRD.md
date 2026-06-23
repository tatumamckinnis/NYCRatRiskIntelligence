# Product Requirements Document — NYC Rat Risk Intelligence

**Version:** 1.0  
**Date:** 2026-06  
**Status:** Shipped (demo / portfolio)

---

## Problem statement

New York City's rodent mitigation budget is fixed. Inspection staff is limited. DOHMH currently dispatches inspectors reactively — when residents complain. Complaint volume is known to be biased by neighborhood income: wealthier neighborhoods complain more, which creates a feedback loop where they receive more inspections regardless of underlying infestation risk.

This project builds a proactive risk-ranking tool: given the week and the neighborhood, predict the probability that an DOHMH inspection will find **active rat signs** — not that residents will complain. This flips the signal from self-reported complaint volume to a direct outcome measure.

---

## Users and personas

### Persona 1 — DOHMH Outreach Coordinator (primary)

> "I plan the next two weeks of proactive inspection routes. I need to know which neighborhoods to prioritize this cycle so we get the best inspection-to-find ratio."

**Needs:**
- A weekly ranked list of high-risk NTAs
- A 12-week forecast so they can plan for seasonal spikes
- Factors driving the score (so they can validate it makes operational sense)
- Export or API access to feed into their routing software

**Frustrations with status quo:**
- Complaint-driven dispatch sends inspectors to well-resourced neighborhoods that complain loudly
- No leading indicator — they respond after infestation is established

**Success metric for them:** Top-decile NTAs contain ≥40% of the week's confirmed active-rat-signs findings.

---

### Persona 2 — Data Journalist / Researcher

> "I'm writing a piece on rat inequality in NYC. I want to understand which neighborhoods have chronic high risk vs. which ones are spiking, and why the model is flagging them."

**Needs:**
- A public-facing map with historical weeks selectable
- Per-NTA breakdowns: risk trend, top contributing factors, nearby restaurant violations
- An exportable API (or at least curl-able JSON) they can cite

**Frustrations:**
- 311 complaint data is easily accessible but a biased signal; they need outcome-based data
- SHAP explanations and factor attributions are buried in CSV files or behind FOIL requests

**Success metric for them:** Can embed a screenshot of the map in an article and cite the methodology in a footnote.

---

### Persona 3 — NYC Resident / Community Organizer

> "My block has been flooded with rats for months. I want to show my City Council member that this is a real, provable problem — not just complaining."

**Needs:**
- A public URL they can share
- Risk score for their NTA, backed by a model trained on inspection outcomes not complaints
- Plain-language explanation of what the score means and what it doesn't mean

**Frustrations:**
- "Just call 311" is the only existing avenue; 311 data is easily dismissed as noise
- No third-party, data-backed framing for advocacy

**Success metric for them:** Risk score ≥ decile 7 for their NTA, shareable on social media.

---

### Persona 4 — Portfolio Reviewer / Recruiter (secondary)

> "I'm evaluating this candidate's ML and full-stack skills. I want to understand what they built, why it's hard, and whether it actually works."

**Needs:**
- Live demo URL that works without a login
- Metrics table they can point to (PR-AUC, lift, RAG benchmarks)
- Readable code and clear architecture
- A case study explaining the non-obvious decisions (label choice, bias analysis, free-tier constraints)

**Frustrations:**
- Projects that list model names without metrics
- Notebooks that only run locally; nothing deployed

**Success metric for them:** Demo loads in <5s, chat answers a question with citations, README explains everything in <10 min.

---

## Key product decisions

### 1. Inspection outcome, not complaint volume, as the label

**Decision:** Train on `DOHMH Rodent Inspections RESULT = 'Active Rat Signs'`, not on 311 call volume.

**Rationale:** 311 complaints are a biased proxy — they reflect who calls 311, not where rats actually are. A model trained on complaints would replicate and amplify that bias. Inspection outcomes are the actual measurement.

**Tradeoff:** Inspection coverage is also not random (inspectors go where they're sent), so there's survivorship bias in the positive label. We address this in the reporting-bias analysis notebook (`ml/notebooks/01_reporting_bias.ipynb`) but do not fully resolve it.

---

### 2. NTA as the spatial unit

**Decision:** Predict at the NTA (Neighborhood Tabulation Area) level, not census tract, block, or address.

**Rationale:** NTA is the smallest unit for which all seven data sources have consistent coverage. Census tracts have too many data gaps in DOB permit data; block-level joins have unacceptable null rates in PLUTO. NTA gives 223 distinct prediction units with stable keys across all data vintages.

**Tradeoff:** NTAs are large (~40k residents on average). Block-level targeting would be more actionable for inspection routing but requires data that isn't reliably available.

---

### 3. BM25-only RAG in production (free tier)

**Decision:** Disable BGE-M3 dense embeddings and BGE Reranker on Render free tier; use BM25 keyword retrieval only for chat in production.

**Rationale:** BGE-M3 requires ~800 MB of RAM to load. Render free tier provides 512 MB. OOM crashes are worse than degraded retrieval quality.

**Tradeoff:** BM25 recall is lower than hybrid (estimated Recall@5 drops from ~0.82 to ~0.68 based on offline eval). Chat answers are still grounded and cited, but may miss relevant statutory sections on complex queries.

**Mitigation:** The `DISABLE_RERANKER` environment variable makes this easy to re-enable on a paid instance. The hybrid path is fully implemented and tested locally.

---

### 4. Groq for LLM generation (not Claude Haiku)

**Decision:** Use Groq (Llama 3.1 70B or similar) as the production LLM for RAG generation. Claude Haiku is used only for development/eval.

**Rationale:** Groq's free tier provides 30 req/min and 6k tokens/min with no credit card required. Claude Haiku requires an Anthropic account with billing set up; the $5 dev credit covers ~500 queries but is not indefinitely renewable for a demo.

**Tradeoff:** Response style is slightly less polished than Claude Haiku. Citations and factual grounding are comparable (both receive the same retrieved chunks in context).

---

## Out of scope

- Authentication / user accounts
- Saved searches or alerts
- Real-time inspection dispatch integration (would require DOHMH system access)
- Address-level or block-level predictions (data coverage too sparse)
- Mapillary street-view segmentation (documented in SPEC cut line)
- AWS ECS multi-cloud deployment (see ADR 0001)
