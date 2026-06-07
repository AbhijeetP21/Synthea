# Clinical Q&A over Synthetic FHIR Data

A clinical question-answering system that uses **retrieval-augmented generation
(RAG)** over synthetic patient records and answers with **every sentence traced
to the FHIR resource it came from**. Built to demonstrate the things clinical AI
teams actually hire for: citation grounding, abstention, PHI governance, and a
CI-gated evaluation harness.

> ⚠️ **Scope & safety.** Synthetic and public data only — **never real PHI**.
> This system is **not clinically validated**, is **not fit for real clinical
> use**, and **does not provide medical advice or treatment recommendations**.
> It is strictly a retrieval-and-grounding demonstrator.

---

## Status

Built in phases (see the build brief). **Phase 1 (MVP) is implemented:**

- ✅ Ingest one Synthea patient → pgvector
- ✅ Patient-scoped semantic retrieval
- ✅ Grounded Q&A with **per-sentence inline citations** (structured, validated output)
- ✅ **Abstention** when evidence is insufficient
- 🔜 Phase 2: hybrid (semantic + keyword/BM25) retrieval
- 🔜 Phase 3: Presidio PHI detection & redaction (seam already in place — `app/ingest/phi.py`)
- 🔜 Phase 4: eval harness (groundedness, hallucination rate, retrieval precision/recall, abstention correctness) + CI gating
- 🔜 Phase 5: Streamlit dashboard with inline citation rendering

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

### The provider seam (`app/llm/`) — swap providers via config, not code

Every LLM and embedding call goes through one module, configured entirely by
environment variables. **No provider is hard-coded anywhere else.**

- **Chat** (`CHAT_PROVIDER`):
  - `anthropic` → Anthropic SDK → **MiniMax-M3** at `https://api.minimax.io/anthropic` (today's default)
  - `openai` → OpenAI SDK → **OpenAI / SynapticaAI / MiniMax OpenAI-mode**
- **Embeddings** (`EMBEDDINGS_PROVIDER`):
  - `local` → **fastembed** (`bge-small-en-v1.5`, 384-dim) — offline, no key, deterministic for CI (default)
  - `openai` → any OpenAI-compatible `/embeddings` endpoint (OpenAI, MiniMax, SynapticaAI)

Chat and embeddings are *separate* seams because the Anthropic SDK has no
embeddings API and embeddings are universally OpenAI-shaped. **Pointing the app
at SynapticaAI later is a base-URL + key change in `.env` — no code change.**

---

## Setup

### Prerequisites
- [mise](https://mise.jdx.dev/) (manages Python + `uv`)
- Docker (Postgres + pgvector, and Synthea generation — no local Java/JDK needed)

### 1. Install the toolchain and dependencies
```powershell
mise install        # Python 3.12 + uv, per mise.toml
mise run install    # uv sync
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

- `source_id` is `<ResourceType>/<resource_id>` — a citation points at the exact
  FHIR resource it came from.
- **Grounding gate** (`app/qa/service.py`): citations that don't resolve to a
  retrieved source are dropped; a sentence with no surviving citation is dropped;
  if nothing grounded remains, the system **abstains** rather than asserting an
  unsupported clinical claim.

---

## Testing
```powershell
mise run test
```
Covers FHIR parsing & code-system mapping, the embeddable-text composition, the
grounding gate (including hallucinated-citation rejection and abstention
fallback), and tolerant JSON parsing.

---

## Guardrails & non-goals
- Not clinically validated; not fit for real use.
- No treatment recommendations or medical advice — reports only what the record states.
- Strictly retrieval + grounding (+ later summarization) over **synthetic** data.
- **Never touches real PHI.**
