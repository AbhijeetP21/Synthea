"""The citation data contract.

A grounded answer is a list of sentences, each carrying one or more citations
that resolve to a specific FHIR resource. Every sentence must cite at least one
source (enforced in app/qa/service.py); abstention is the explicit alternative.

``source_id`` is a stable reference of the form ``<ResourceType>/<resource_id>``
(e.g. ``MedicationRequest/abc-123``). In the MVP one chunk maps to one resource,
so a citation points directly at the resource it was drawn from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Coding(BaseModel):
    """A clinical code surfaced in a citation (LOINC / RxNorm / SNOMED CT)."""

    system: str | None = None  # e.g. "http://loinc.org"
    system_name: str | None = None  # human label: "LOINC", "RxNorm", "SNOMED CT"
    code: str | None = None
    display: str | None = None


class Source(BaseModel):
    """A retrievable, citable unit derived from one FHIR resource."""

    source_id: str  # "<ResourceType>/<resource_id>"
    resource_type: str
    resource_id: str
    patient_id: str
    title: str  # short human label, e.g. "Medication: Lisinopril 10 MG Oral Tablet"
    text: str  # the chunk text that was embedded / shown to the model
    date: str | None = None
    codings: list[Coding] = Field(default_factory=list)


class AnswerSentence(BaseModel):
    """One sentence of the answer plus the sources backing it."""

    text: str
    citations: list[str] = Field(default_factory=list)  # source_ids


class LLMAnswer(BaseModel):
    """Raw structured output expected from the model (before resolution)."""

    abstained: bool = False
    abstention_reason: str | None = None
    sentences: list[AnswerSentence] = Field(default_factory=list)


class AskRequest(BaseModel):
    patient_id: str
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    """What the API returns: the answer, plus every cited source resolved in full
    so the dashboard can render citations inline."""

    patient_id: str
    question: str
    abstained: bool
    abstention_reason: str | None = None
    sentences: list[AnswerSentence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)  # only those actually cited
