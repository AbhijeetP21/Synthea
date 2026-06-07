"""FastAPI service exposing the clinical Q&A endpoints.

Endpoints:
  GET  /health        liveness
  GET  /patients      patient_ids currently ingested
  POST /ask           grounded, citation-bearing answer (or abstention)
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from app.db import list_patients
from app.qa.service import QAService
from app.schemas import AskRequest, AskResponse

app = FastAPI(
    title="Clinical Q&A (RAG over synthetic FHIR)",
    version="0.1.0",
    description=(
        "Citation-grounded question answering over synthetic patient records. "
        "Synthetic data only; not clinically validated; provides no medical advice."
    ),
)


@lru_cache
def _service() -> QAService:
    """Lazily constructed so importing the app (e.g. in tests) does not require
    a live database or embedding model."""
    return QAService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/patients")
def patients() -> dict:
    return {"patients": list_patients()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _service().ask(req.patient_id, req.question)
