# FitCheck AI

Paste a resume and a target job description. An LLM reads both, reasons about fit requirement-by-requirement (matched / partial / missing, with evidence and reasoning for every single one — including matches), and suggests bullet rewrites grounded only in experience you actually have. A RAG layer over a curated 32-skill knowledge base grounds the "how to close this gap" suggestions so the model can't invent a course or certification name.

Live browser demo: `static/demo.html` (real LLM calls via Groq's free API — paste a key in the page, see below). Full pipeline: `src/`, orchestrated with LangGraph.

> **Note on this fork:** originally built against the Anthropic API. Migrated to Groq (`openai/gpt-oss-120b`, called through the OpenAI-compatible SDK/endpoint) so it runs on Groq's free tier — no credit card, get a key at [console.groq.com/keys](https://console.groq.com/keys). All reasoning/architecture below is unchanged; only the LLM client and env var name (`GROQ_API_KEY`) changed.

## Why this is genuinely AI-centric, not AI-flavored

A previous version of this idea (a government-scheme eligibility matcher) used an LLM only to phrase pre-computed, rule-based results — accurate, but the AI was decoration on top of `if`/`else`. This project is built the opposite way on purpose:

- **Extraction** (`src/extraction.py`): turning "Built and shipped a RAG pipeline with a 15-case eval harness, improved retrieval MRR from 0.83 to 1.0" into structured skills isn't a rule — it requires reading the sentence.
- **Matching** (`src/matcher.py`): deciding whether "led a cross-functional analytics initiative" satisfies "stakeholder management experience" requires reasoning about equivalence, not keyword overlap. There is no offline fallback for this step, deliberately — see the error message if you run it without a key.
- **Report generation** (`src/report.py`): rewriting a resume bullet to better match JD language is a generation task with a hard constraint (never fabricate), not a template fill.

The only part that is *not* an LLM call is the "where do I go to close this gap" lookup, and that's intentional the other direction: retrieval (BM25 over `data/skill_resources.json`) supplies real, pre-written resources so the model is never asked to invent a course name from memory — the same discipline as before, applied to the one place hallucination would be easy and hard to notice.

## Architecture (LangGraph)

```
START -> extract_resume -> extract_jd -> match --+-- score < 25 --> low_fit_advisory -> END
                                                   +-- score >= 25 --> generate_report -> END
```

The conditional edge is the reason this is a graph and not just four chained function calls: a genuinely poor fit doesn't get the same optimistic bullet-rewrite treatment as a borderline one — it gets routed to a node that says so plainly instead of manufacturing false hope. See `src/graph.py`.

Every LLM call uses Groq's **Structured Outputs, strict mode** (`response_format: {type:"json_schema", json_schema:{strict:true,...}}`) — not "please respond in JSON" and not tool-calling either (Groq's docs: tool use isn't supported with Structured Outputs, so a "strict" flag on a tool definition is silently ignored -- see `src/_groq_strict.py` docstring for the full story of finding that out the hard way). The Pydantic models in `src/schemas.py` generate the actual schema, so there's one source of truth for the data shape instead of a schema that quietly drifts from a regex parser somewhere. Strict mode uses real constrained decoding: the model is restricted at the token level, so a missing or renamed field is structurally impossible, not just prompted against.

## The honest constraint this was built under

Same sandbox as before: no route to an LLM API key, no route to huggingface.co. This time that constraint bites harder, because the core value of this project *is* the LLM reasoning — there's no defensible rule-based fallback to fall back to (and building a fake one would defeat the entire point you pushed back on). So:

- `src/extraction.py`, `src/matcher.py`, `src/report.py` are written, schema-validated, and structurally correct, but **not run end-to-end in this environment** — they raise a clear `RuntimeError` rather than silently degrading. `eval/evaluate.py::eval_llm_steps()` is a stub with the exact methodology for evaluating them written out, not run.
- `static/demo.html` needs a key too now (Groq doesn't have Anthropic-artifact's browser-key-free trick) — paste a free Groq key into the field at the top of the page. It calls `api.groq.com` directly from your browser; the key stays in that page's memory for the session, never sent anywhere else, never stored.
- Everything that *can* be tested without a key **was** tested, the same way as before — see next section.

## What's actually tested (22/22)

```
python3 -m eval.evaluate
```

- **Resource retrieval (RAG layer)**: 15/15 hand-verified test cases (`eval/resource_retrieval_tests.json`) — free-text gap descriptions correctly retrieve the matching skill entry.
- **Anti-fabrication grounding check**: 7/7 (`eval/grounding_check_tests.json`) — a second, independent, mechanical filter (token-overlap against the source resume, not just LLM self-restraint) that catches a hallucinated `based_on` before it reaches the user. Found and fixed two real bugs building this: a punctuation-tokenizing bug (`scikit-learn.` ≠ `learn` as tokens because the sentence-ending period glued on) and a threshold calibrated too high for genuine paraphrases (fixed by measuring the real separation: grounded claims scored 0.43–0.57 overlap, fabricated ones scored 0.00 — recalibrated to 0.5 with margin in both directions).
- The `static/demo.html` grounding check is a simplified JS mirror (no stemmer) with its threshold recalibrated separately (0.4) against the same measured gap.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...              # free, no credit card -- console.groq.com/keys
                                          # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
python3 -m eval.evaluate                 # the 22/22 that runs without a key
python3 -m src.graph                     # prints graph structure without a key; runs the real pipeline with one
streamlit run app.py                     # full interactive app
```

For `static/demo.html`: just open the file in any browser (double-click it, or `open static/demo.html` / `start static/demo.html`) and paste your Groq key into the field at the top of the page. No server, no build step.

**Model:** defaults to `openai/gpt-oss-120b` on Groq -- a current production model on the free tier that supports Structured Outputs strict mode (only `openai/gpt-oss-20b` and `openai/gpt-oss-120b` do, as of this writing). Set the `model=` argument in any of `extract_profile()` / `match_profiles()` / `generate_report()` (or `GROQ_MODEL` in `demo.html`) to swap models; check [console.groq.com/docs/structured-outputs](https://console.groq.com/docs/structured-outputs) for current strict-mode model support before swapping, since it's more limited than general model availability.

## Project layout

```
data/skill_resources.json     32-skill curated knowledge base (RAG corpus, retrieval-only)
src/schemas.py                Pydantic models = the tool input_schemas the LLM is constrained to
src/extraction.py             unstructured text -> structured profile (LLM, Structured Outputs strict mode)
src/matcher.py                requirement-by-requirement fit reasoning (LLM, Structured Outputs strict mode)
src/resource_rag.py           BM25 retrieval over the skill knowledge base (no LLM)
src/report.py                 bullet rewrites + mechanical anti-fabrication filter
src/graph.py                  LangGraph orchestration with the low-fit conditional branch
eval/                         everything testable without a key, run for real; LLM-eval methodology documented, not faked
app.py                        Streamlit app (needs a key)
static/demo.html              live browser demo (real Groq calls, paste a free key into the page)
```

## Next steps

1. Hand-label 10-15 (resume, JD) pairs and run `eval_llm_steps()` for real extraction/matching precision once you have a key.
2. Expand the skill knowledge base past 32 entries, or make it category-specific (e.g. a version scoped to DS/MLE roles vs. general SWE).
3. Add a second LLM pass that critiques its own match reasoning before returning it (an actual self-correction loop, which LangGraph makes easy to add as one more conditional edge back into `match`).
4. Deploy `app.py` (Streamlit Community Cloud / HF Spaces) for a shareable link people can actually use on their own resumes.
