"""Shared fixtures: a tiny synthetic FHIR bundle covering each resource type."""

from __future__ import annotations

import pytest


@pytest.fixture
def mini_bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": "1972-04-10",
                    "name": [{"given": ["Jane"], "family": "Synthetic"}],
                    "address": [{"city": "Boston", "state": "MA"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "subject": {"reference": "Patient/pat-1"},
                    "clinicalStatus": {"text": "active"},
                    "onsetDateTime": "2019-06-01T00:00:00Z",
                    "code": {
                        "text": "Essential hypertension",
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "59621000",
                                "display": "Essential hypertension",
                            }
                        ],
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-1",
                    "status": "active",
                    "authoredOn": "2020-01-15",
                    "subject": {"reference": "Patient/pat-1"},
                    "medicationCodeableConcept": {
                        "text": "Lisinopril 10 MG Oral Tablet",
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "314076",
                                "display": "Lisinopril 10 MG Oral Tablet",
                            }
                        ],
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "effectiveDateTime": "2021-03-03",
                    "subject": {"reference": "Patient/pat-1"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "8867-4",
                                "display": "Heart rate",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 72, "unit": "/min"},
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-1",
                    "criticality": "high",
                    "patient": {"reference": "Patient/pat-1"},
                    "code": {
                        "text": "Penicillin",
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "373270004",
                                "display": "Penicillin",
                            }
                        ],
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "subject": {"reference": "Patient/pat-1"},
                    "period": {"start": "2021-03-03T09:00:00Z"},
                    "type": [{"text": "General examination"}],
                }
            },
            {
                # Unsupported type — must be ignored by the parser.
                "resource": {"resourceType": "Organization", "id": "org-1", "name": "Acme"}
            },
        ],
    }
