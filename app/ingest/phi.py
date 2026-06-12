"""PHI detection & redaction — the data-governance stage (Stage 3).

Every piece of text is run through this before it is embedded, persisted, or
sent to the model, so no un-redacted identifier reaches the vector store or the
LLM. Detection uses Microsoft Presidio (spaCy NER + rule-based recognizers),
which maps onto the HIPAA Safe Harbor identifier categories.

Two design choices worth calling out:

* **Structured hints.** FHIR is structured, so at ingest time we already know
  the patient's name and address. Synthea's names carry numeric suffixes
  ("Vanna750 Rosenbaum794") that defeat NER, so we feed the known identifier
  strings to Presidio as a deny-list. De-identification therefore does not
  depend on the model *guessing* that a token is a name.
* **Detect-all, redact-curated.** We *report* every identifier category we find
  (the governance signal) but only *redact* direct identifiers. Dates are
  reported and kept by default, because onset/authoring dates are clinical
  content here and the data is synthetic — `PHI_REDACT_DATES=true` flips that.

Synthetic data only — this stage demonstrates governance, it is not a license to
process real PHI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache

from pydantic import BaseModel

from app.config import get_settings

# Direct identifiers we redact (HIPAA Safe Harbor: names, geography, contacts,
# SSNs, account / license / device / vehicle numbers, URLs, IPs, biometrics).
REDACT_ENTITIES = frozenset({
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "MRN",
    "LOCATION", "URL", "IP_ADDRESS", "US_DRIVER_LICENSE", "MEDICAL_LICENSE",
    "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE", "US_PASSPORT", "CRYPTO",
})

# Detected and reported, but kept unless PHI_REDACT_DATES is set.
DATE_ENTITIES = frozenset({"DATE_TIME", "AGE"})


class RedactionResult(BaseModel):
    """Redacted text plus what was found (all detections) and what was removed."""

    text: str
    detected: dict[str, int] = {}
    redacted: dict[str, int] = {}


class Redactor(ABC):
    @abstractmethod
    def redact(self, text: str, *, hints: dict[str, list[str]] | None = None) -> RedactionResult:
        """Redact `text`. `hints` maps an entity type to known identifier strings
        (e.g. {"PERSON": ["Jane Synthetic"]}) to seed deny-list recognizers."""


class NoOpRedactor(Redactor):
    """Pass-through, for when PHI redaction is disabled or in offline tests."""

    def redact(self, text: str, *, hints: dict[str, list[str]] | None = None) -> RedactionResult:
        return RedactionResult(text=text)


class PresidioRedactor(Redactor):
    def __init__(
        self,
        *,
        redact_entities: frozenset[str] | set[str],
        score_threshold: float,
        model: str,
        language: str = "en",
    ) -> None:
        self._redact = set(redact_entities)
        self._threshold = score_threshold
        self._model = model
        self._language = language
        self._analyzer = None  # built lazily — the spaCy model is heavy.
        self._anonymizer = None

    def _ensure_engines(self) -> None:
        if self._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": self._language, "model_name": self._model}],
            }
        ).create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=[self._language]
        )
        # Custom rule recognizers — the "rules" half of NER+rules. These also
        # patch this Presidio build, where US_SSN does not fire on dashed SSNs
        # and there is no medical-record-number recognizer at all.
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="US_SSN",
                patterns=[Pattern("ssn-dashed", r"\b\d{3}-\d{2}-\d{4}\b", 0.85)],
            )
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="MRN",
                patterns=[Pattern("mrn", r"\b\d{6,10}\b", 0.4)],
                context=["mrn", "medical record", "record number"],
            )
        )
        self._analyzer = analyzer
        self._anonymizer = AnonymizerEngine()

    def _ad_hoc(self, hints: dict[str, list[str]] | None):
        if not hints:
            return None
        from presidio_analyzer import PatternRecognizer

        recognizers = []
        for entity, terms in hints.items():
            deny = sorted({t for t in terms if t and len(t) > 1})
            if deny:
                recognizers.append(
                    PatternRecognizer(supported_entity=entity, deny_list=deny)
                )
        return recognizers or None

    def redact(self, text: str, *, hints: dict[str, list[str]] | None = None) -> RedactionResult:
        if not text:
            return RedactionResult(text=text)
        self._ensure_engines()
        results = self._analyzer.analyze(
            text=text,
            language=self._language,
            score_threshold=self._threshold,
            ad_hoc_recognizers=self._ad_hoc(hints),
        )
        detected = Counter(r.entity_type for r in results)
        to_redact = [r for r in results if r.entity_type in self._redact]
        if to_redact:
            text = self._anonymizer.anonymize(text=text, analyzer_results=to_redact).text
        return RedactionResult(
            text=text,
            detected=dict(detected),
            redacted=dict(Counter(r.entity_type for r in to_redact)),
        )


@lru_cache
def get_redactor() -> Redactor:
    settings = get_settings()
    if not settings.phi_redaction:
        return NoOpRedactor()
    redact = set(REDACT_ENTITIES)
    if settings.phi_redact_dates:
        redact |= DATE_ENTITIES
    return PresidioRedactor(
        redact_entities=redact,
        score_threshold=settings.phi_score_threshold,
        model=settings.phi_spacy_model,
    )
