from app.ingest.fhir_parser import parse_bundle
from app.ingest.pipeline import _embeddable_text


def test_embeddable_text_appends_clinical_codes(mini_bundle):
    _, resources = parse_bundle(mini_bundle)
    med = next(r for r in resources if r.resource_type == "MedicationRequest")
    text = _embeddable_text(med)
    # Exact drug name and RxNorm code are both present for retrieval.
    assert "Lisinopril" in text
    assert "RxNorm 314076" in text
