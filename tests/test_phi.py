"""PHI stage (Phase 3). The Presidio test loads the spaCy model once (module
scope) — it is the one slow test, kept isolated so the rest of the suite stays
fast. Verifies identifiers are removed while clinical content survives."""

import pytest

from app.ingest.fhir_parser import collect_phi_hints
from app.ingest.phi import REDACT_ENTITIES, NoOpRedactor, PresidioRedactor


def test_noop_redactor_passthrough():
    r = NoOpRedactor().redact("Patient Jane Synthetic, hypertension.")
    assert r.text == "Patient Jane Synthetic, hypertension."
    assert r.detected == {} and r.redacted == {}


def test_collect_phi_hints_from_structured_patient(mini_bundle):
    hints = collect_phi_hints(mini_bundle)
    assert "Jane" in hints["PERSON"]
    assert "Synthetic" in hints["PERSON"]
    assert "Jane Synthetic" in hints["PERSON"]
    assert "Boston" in hints["LOCATION"]
    assert "MA" in hints["LOCATION"]


@pytest.fixture(scope="module")
def redactor():
    return PresidioRedactor(
        redact_entities=REDACT_ENTITIES, score_threshold=0.6, model="en_core_web_lg"
    )


def test_redacts_identifiers_keeps_clinical_content(redactor):
    text = (
        "Patient Vanna750 Rosenbaum794. Email a@b.com, phone (617) 555-0142. "
        "Condition (diagnosis): Prediabetes (finding). [RxNorm 314076]"
    )
    hints = {"PERSON": ["Vanna750", "Rosenbaum794"]}
    res = redactor.redact(text, hints=hints)

    # Identifiers gone.
    assert "Vanna750" not in res.text and "Rosenbaum794" not in res.text
    assert "a@b.com" not in res.text
    assert "617" not in res.text
    assert "<PERSON>" in res.text and "<EMAIL_ADDRESS>" in res.text
    # Clinical content preserved verbatim.
    assert "Prediabetes (finding)" in res.text
    assert "RxNorm 314076" in res.text
    # Reported as redacted.
    assert res.redacted.get("PERSON", 0) >= 1
    assert "EMAIL_ADDRESS" in res.redacted


def test_dates_detected_but_kept_by_default(redactor):
    res = redactor.redact("Onset 2004-01-16 of essential hypertension.")
    # DATE_TIME is flagged...
    assert res.detected.get("DATE_TIME", 0) >= 1
    # ...but not redacted (clinical content here), and not in the redact set.
    assert "2004-01-16" in res.text
    assert "DATE_TIME" not in res.redacted
    assert "DATE_TIME" not in REDACT_ENTITIES
