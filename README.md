# Clinical Q&A over Synthetic FHIR Data

A clinical question-answering system that uses **retrieval-augmented generation
(RAG)** over synthetic patient records and answers with **every sentence traced
to the FHIR resource it came from**. Built to demonstrate the things clinical AI
teams actually hire for: citation grounding, abstention, PHI governance, and a
CI-gated evaluation harness.

> **Scope & safety.** Synthetic and public data only - **never real PHI**.
> This system is **not clinically validated**, is **not fit for real clinical
> use**, and **does not provide medical advice or treatment recommendations**.
> It is strictly a retrieval-and-grounding demonstrator.

---

## Status

Built in stages (see the build brief). **All five stages are implemented:**

- **Stage 1 (MVP):** ingest one Synthea patient → pgvector; patient-scoped
  semantic retrieval; grounded Q&A with **per-sentence inline citations**
  (structured, validated output)
- **Stage 2 (Abstention):** refuses correctly when evidence is insufficient -
  including off-topic / out-of-record questions - via a retrieval **relevance
  gate** plus the grounding gate (see below)
- **Stage 3 (PHI):** Presidio-based detection & redaction of HIPAA Safe Harbor
  identifiers, before any text is embedded, stored, or sent to the model (see below)
- **Stage 4 (Eval harness + CI gate):** labeled gold set scored on groundedness,
  hallucination rate, retrieval precision/recall, and abstention correctness;
  deterministic metrics gate every PR in CI (see below)
- **Stage 5 (Dashboard):** Streamlit UI over the API that renders every answer
  with **inline, clickable citations** to the FHIR sources (see below)

> *Hybrid (semantic + keyword/BM25) retrieval - §6 of the brief - is built behind
> the `Retriever` seam as a later increment; semantic-only is the MVP baseline.*

---

## Architecture

```
Synthea FHIR bundle
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ Ingestion (app/ingest)                                     │
│   parse FHIR → PHI redact (hook) → embed → store           │
└──────────────────────────────────────────────────────────┘
   │                                            ▲
   ▼                                            │ embeddings
┌─────────────────────┐                  ┌──────────────────┐
│ pgvector (app/db)   │◄─── retrieve ────│ Provider seam    │
└─────────────────────┘                  │ (app/llm)        │
   │                                      │  chat: MiniMax / │
   ▼                                      │   OpenAI / Synap │
┌─────────────────────┐    grounded      │  embed: local /  │
│ Q&A (app/qa)        │───  answer  ─────▶│   OpenAI-compat  │
│  retrieve→prompt→   │                   └──────────────────┘
│  enforce grounding  │
└─────────────────────┘
   │
   ▼
FastAPI /ask  →  per-sentence citations + resolved sources
```

### The provider seam (`app/llm/`) - swap providers via config, not code

Every LLM and embedding call goes through one module, configured entirely by
environment variables. **No provider is hard-coded anywhere else.**

- **Chat** (`CHAT_PROVIDER`):
  - `anthropic` → Anthropic SDK → **MiniMax-M3** at `https://api.minimax.io/anthropic` (today's default)
  - `openai` → OpenAI SDK → **OpenAI / SynapticaAI / MiniMax OpenAI-mode**
- **Judge** (`JUDGE_PROVIDER`): the eval LLM-as-judge, on its own seam so it can
  be a *different vendor and model family* than the generator - here OpenAI
  `gpt-5.5` grading MiniMax, so the system is not marking its own work. The
  OpenAI adapter adapts to newer-model parameter rules (`max_completion_tokens`,
  fixed temperature) automatically, so the swap stays config-only.
- **Embeddings** (`EMBEDDINGS_PROVIDER`):
  - `local` → **fastembed** (`bge-small-en-v1.5`, 384-dim) - offline, no key, deterministic for CI (default)
  - `openai` → any OpenAI-compatible `/embeddings` endpoint (OpenAI, MiniMax, SynapticaAI)

Chat and embeddings are *separate* seams because the Anthropic SDK has no
embeddings API and embeddings are universally OpenAI-shaped. **Pointing the app
at SynapticaAI later is a base-URL + key change in `.env` - no code change.**

---

## Setup

### Prerequisites
- [mise](https://mise.jdx.dev/) (manages Python + `uv`)
- Docker (Postgres + pgvector, and Synthea generation - no local Java/JDK needed)

### 1. Install the toolchain and dependencies
```powershell
mise install        # Python 3.12 + uv, per mise.toml
mise run install    # uv sync - also pulls the spaCy en_core_web_lg model (~400MB) for PHI
```

### 2. Configure environment
```powershell
Copy-Item .env.example .env
# Edit .env: set CHAT_API_KEY to your MiniMax key. Defaults use local embeddings
# (no key needed) so ingestion/retrieval work offline.
```

### 3. Start the datastore
```powershell
docker compose up -d db
```

### 4. Generate a synthetic patient and ingest
```powershell
mise run synthea                                   # writes data/sample/patient_bundle.json
mise run ingest data/sample/patient_bundle.json
```

### 5. Run the API
```powershell
mise run api        # http://localhost:8000/docs
```

Ask a question:
```powershell
$body = @{ patient_id = "<id printed by ingest>"; question = "What medications is this patient on, and any flagged allergies?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType application/json -Body $body
```

### 6. Run the dashboard
```powershell
mise run dashboard  # http://localhost:8501  (keep the API from step 5 running)
```

### Run everything in containers (demo)
```powershell
docker compose --profile app up
```

---

## The citation data contract

Answers are **structured, not free text**, and validated with Pydantic
(`app/schemas.py`):

```jsonc
{
  "abstained": false,
  "sentences": [
    { "text": "The patient is on Lisinopril 10 MG Oral Tablet.",
      "citations": ["MedicationRequest/med-1"] }
  ],
  "sources": [ /* every cited source, resolved in full for inline rendering */ ]
}
```

- `source_id` is `<ResourceType>/<resource_id>` - a citation points at the exact
  FHIR resource it came from.
- **Grounding gate** (`app/qa/service.py`): citations that don't resolve to a
  retrieved source are dropped; a sentence with no surviving citation is dropped;
  if nothing grounded remains, the system **abstains** rather than asserting an
  unsupported clinical claim.

---

## Abstention (Stage 2)

The system refuses to answer when the record doesn't support an answer. Two
independent gates make this robust:

1. **Relevance gate** (`app/retrieval/retriever.py`): semantic search returns
   each chunk's cosine distance, and chunks beyond `RETRIEVAL_MAX_DISTANCE` are
   discarded. An off-topic or out-of-record question (*"What is the capital of
   France?"*) finds nothing close enough, retrieves zero evidence, and the
   service abstains **without ever calling the model**. The threshold (default
   `0.40`) was calibrated on the ingested patient - relevant queries land at
   cosine distance ~0.24-0.34, off-topic ones at ~0.42+ - and is env-tunable so
   the Stage 4 eval harness can optimize it.
2. **Grounding gate** (above): even when evidence *is* retrieved, any answer the
   model can't tie back to it is dropped, falling through to abstention.

The model is also instructed to abstain on insufficient evidence and to answer
only the supported parts of a multi-part question. Net effect: the system says
*"No matching information was found in this patient's record"* instead of
guessing - the strongest safety signal the brief asks for.

---

## PHI detection & redaction (Stage 3)

Before any text is embedded, stored, or sent to the model, it passes through the
PHI stage (`app/ingest/phi.py`), which uses **Microsoft Presidio** (spaCy NER +
rule-based recognizers) to find HIPAA Safe Harbor identifiers. Two deliberate
design choices:

- **Structured hints, not NER guesswork.** FHIR is structured, so at ingest time
  we already know the patient's name and address. Synthea's names carry numeric
  suffixes (*"Vanna750 Rosenbaum794"*) that defeat NER outright - so
  `collect_phi_hints()` pulls the known identifier strings from the Patient
  resource and seeds them as a Presidio deny-list. De-identification therefore
  does not depend on the model *inferring* that a token is a name. Custom
  recognizers cover the gaps in the base library (dashed SSNs, medical record
  numbers).
- **Detect-all, redact-curated.** Every identifier category found is *reported*
  (the governance signal - `mise run ingest` prints the tally); direct
  identifiers (names, geography, contacts, SSN/MRN, account/license numbers) are
  *redacted* to `<PERSON>`, `<LOCATION>`, etc. **Dates are flagged but kept by
  default** (`PHI_REDACT_DATES=false`): onset and authoring dates are clinical
  content that powers the grounded answers, and the data is synthetic. This is
  the Safe-Harbor trade-off made explicit rather than blindly nuking dates.

After redaction the stored Patient record reads
`Patient <PERSON>. Gender: female. Date of birth: 1966-10-28. Address: <LOCATION>, <LOCATION>.`
- the name and city are gone everywhere, while gender, dates, and all clinical
codes survive so retrieval and citations are unaffected. Disable the stage with
`PHI_REDACTION=false`.

> Synthetic data only. This stage demonstrates data governance; it is **not** a
> license to process real PHI.

---

## Eval harness + CI gate (Stage 4)

A labeled **gold set** (`evals/gold_set.yaml`, ~14 items against the ingested
patient) drives the four metrics the brief asks for. The harness runs in two
layers:

- **Deterministic layer (no API key)** - *retrieval precision/recall* against
  labeled `source_id`s, and *abstention reachability*: off-topic questions must
  retrieve nothing (so the relevance gate guarantees abstention), answerable ones
  must retrieve evidence. This layer is the **hard CI merge gate** - it runs on a
  Postgres+pgvector service container on every PR (`.github/workflows/ci.yml`),
  so a retrieval or prompt change that regresses is caught before merge.
- **LLM layer (needs a key)** - runs the real Q&A path for true *abstention
  correctness* and a `must_include` content check, plus an **LLM-as-judge**
  (`evals/judge.py`) scoring *groundedness* and *hallucination rate* sentence by
  sentence against each sentence's cited evidence. Advisory by default; `--strict`
  makes it gate too.

```powershell
mise run eval                              # full run if CHAT_API_KEY is set
uv run python -m evals.run --no-llm        # deterministic gate only (what CI runs)
```

Latest local run - **MiniMax-M3 generator graded by an independent
`gpt-5.5` judge** (different vendor and model family), 14 items:

| Metric | Score |
|---|---|
| Retrieval recall | **1.00** |
| Retrieval precision | 0.23 *(small relevant sets vs. top-8 - recall is the safety-relevant number)* |
| Abstention reachability (deterministic) | **1.00** |
| Abstention correctness (LLM path) | **1.00** *(abstains on blood type, cancer hx, procedures, vaccinations, redacted contact info, off-topic)* |
| Groundedness | **0.92** |
| Hallucination rate | **0.08** |

The non-perfect groundedness is the harness working as intended. The
independent judge flagged one sentence on the BMI question: the model correctly
read the cited BMI *value* (28.32 kg/m2) from the Observation, but then asserted
it was "below the 30 kg/m2 obesity threshold" - and that 30 cutoff appears
*nowhere in the cited resource*. The model imported a fact from its own training
and presented it as cited. A weaker, same-family judge waved this through; the
stronger independent judge caught the unstated clinical assumption. That is
exactly the failure mode an eval harness exists to surface.

> Honest caveats: the gold set is small (14 items). The grounding gate drops
> *uncited* sentences before the judge ever runs, so the judge's job is the
> subtler one above - catching a *cited* sentence whose citation doesn't actually
> support it. The judge runs on its own swappable `JUDGE_*` seam (here OpenAI
> `gpt-5.5`, independent of the MiniMax generator). Two distinct abstention
> mechanisms are measured separately: the deterministic relevance gate
> (off-topic) and the grounded LLM layer (clinically-adjacent but out-of-record).

---

## Dashboard (Stage 5)

A Streamlit UI (`streamlit_app.py`) talks to the FastAPI service over HTTP - so
it shows exactly what any client would get - and renders the traceability that
is the whole point of the system:

- Each answer sentence carries **superscript `[n]` citations that link down to
  numbered source cards** (the resolved FHIR resource: its text, clinical codes,
  and date). Grounding is *visible*, not just claimed.
- **Abstention is shown as a deliberate notice**, not an error - the safety
  behavior reads as a feature.
- Redacted tokens (`<PERSON>`, `<LOCATION>`) appear in the source text, so the
  PHI stage is visible too.

```powershell
mise run api          # terminal 1
mise run dashboard    # terminal 2 -> http://localhost:8501
```

The pure rendering logic (citation numbering, HTML escaping of redaction tokens)
lives in `app/web/render.py`, separate from the Streamlit script so it is unit
tested.

---

## Testing
```powershell
mise run test
```
Covers FHIR parsing & code-system mapping, the embeddable-text composition, the
grounding gate (including hallucinated-citation rejection and abstention
fallback), tolerant JSON parsing, PHI redaction, eval metrics, and the dashboard
rendering helpers.

---

## Guardrails & non-goals
- Not clinically validated; not fit for real use.
- No treatment recommendations or medical advice - reports only what the record states.
- Strictly retrieval + grounding (+ later summarization) over **synthetic** data.
- **Never touches real PHI.**
