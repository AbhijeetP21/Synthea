"""PHI detection & redaction seam.

Phase 1 ships a no-op pass-through so the pipeline shape is final and ingestion
runs end-to-end. Phase 3 replaces ``NoOpRedactor`` with a Presidio-backed
implementation that detects/redacts the 18 HIPAA Safe Harbor identifiers
*before any text reaches the model or the vector store*. Keeping the seam here
means that swap touches one class, not the pipeline.

Synthetic data only — this stage demonstrates governance, it is not a license to
process real PHI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Redactor(ABC):
    @abstractmethod
    def redact(self, text: str) -> str: ...


class NoOpRedactor(Redactor):
    """Phase 1 placeholder. Returns text unchanged."""

    def redact(self, text: str) -> str:
        return text


def get_redactor() -> Redactor:
    # Phase 3 will branch here on config to return a PresidioRedactor.
    return NoOpRedactor()
