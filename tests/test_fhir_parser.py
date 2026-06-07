from app.ingest.fhir_parser import parse_bundle


def test_parses_only_supported_resources(mini_bundle):
    patient_id, resources = parse_bundle(mini_bundle)
    assert patient_id == "pat-1"

    types = {r.resource_type for r in resources}
    assert types == {
        "Patient",
        "Condition",
        "MedicationRequest",
        "Observation",
        "AllergyIntolerance",
        "Encounter",
    }
    # Organization is unsupported and must be dropped.
    assert "Organization" not in types
    assert all(r.patient_id == "pat-1" for r in resources)


def test_source_ids_are_stable_references(mini_bundle):
    _, resources = parse_bundle(mini_bundle)
    by_type = {r.resource_type: r for r in resources}
    assert by_type["MedicationRequest"].source_id == "MedicationRequest/med-1"
    assert by_type["Condition"].source_id == "Condition/cond-1"


def test_code_systems_are_named(mini_bundle):
    _, resources = parse_bundle(mini_bundle)
    by_type = {r.resource_type: r for r in resources}

    med = by_type["MedicationRequest"]
    assert med.codings[0].system_name == "RxNorm"
    assert med.codings[0].code == "314076"

    cond = by_type["Condition"]
    assert cond.codings[0].system_name == "SNOMED CT"

    obs = by_type["Observation"]
    assert obs.codings[0].system_name == "LOINC"


def test_renders_human_readable_text(mini_bundle):
    _, resources = parse_bundle(mini_bundle)
    med = next(r for r in resources if r.resource_type == "MedicationRequest")
    assert "Lisinopril" in med.text
    assert med.date == "2020-01-15"
