"""Parse a Synthea FHIR R4 bundle into citable, render-ready resource records.

We touch exactly the resource types the brief calls for — Patient, Condition,
MedicationRequest, Observation, AllergyIntolerance, Encounter — and surface the
clinical code systems (LOINC, RxNorm, SNOMED CT) as first-class metadata so they
can appear in citations.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas import Coding

# Map FHIR code-system URIs to the human names clinicians expect in citations.
CODE_SYSTEMS: dict[str, str] = {
    "http://loinc.org": "LOINC",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://snomed.info/sct": "SNOMED CT",
    "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10-CM",
    "http://unitsofmeasure.org": "UCUM",
}

SUPPORTED_TYPES = {
    "Patient",
    "Condition",
    "MedicationRequest",
    "Observation",
    "AllergyIntolerance",
    "Encounter",
}


class ParsedResource:
    """A single FHIR resource normalized for embedding and citation."""

    def __init__(
        self,
        *,
        resource_type: str,
        resource_id: str,
        patient_id: str,
        title: str,
        text: str,
        date: str | None = None,
        codings: list[Coding] | None = None,
    ) -> None:
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.patient_id = patient_id
        self.title = title
        self.text = text
        self.date = date
        self.codings = codings or []

    @property
    def source_id(self) -> str:
        return f"{self.resource_type}/{self.resource_id}"


def load_bundle(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _codings(concept: dict | None) -> list[Coding]:
    if not concept:
        return []
    out: list[Coding] = []
    for c in concept.get("coding", []):
        system = c.get("system")
        out.append(
            Coding(
                system=system,
                system_name=CODE_SYSTEMS.get(system, system),
                code=c.get("code"),
                display=c.get("display"),
            )
        )
    return out


def _concept_text(concept: dict | None) -> str:
    if not concept:
        return "unknown"
    if concept.get("text"):
        return concept["text"]
    for c in concept.get("coding", []):
        if c.get("display"):
            return c["display"]
    return "unknown"


def _date(resource: dict, *keys: str) -> str | None:
    for k in keys:
        val = resource.get(k)
        if isinstance(val, str):
            return val[:10]
        if isinstance(val, dict) and val.get("start"):
            return val["start"][:10]
    return None


# --- per-resource renderers -------------------------------------------------


def _render_patient(r: dict) -> tuple[str, str, list[Coding]]:
    name = ""
    if r.get("name"):
        n = r["name"][0]
        given = " ".join(n.get("given", []))
        name = f"{given} {n.get('family', '')}".strip()
    gender = r.get("gender", "unknown")
    birth = r.get("birthDate", "unknown")
    addr = ""
    if r.get("address"):
        a = r["address"][0]
        addr = f"{a.get('city', '')}, {a.get('state', '')}".strip(", ")
    title = f"Patient: {name or r.get('id')}"
    text = (
        f"Patient {name}. Gender: {gender}. Date of birth: {birth}. "
        f"Address: {addr or 'unknown'}."
    )
    return title, text, []


def _render_condition(r: dict) -> tuple[str, str, list[Coding]]:
    label = _concept_text(r.get("code"))
    codings = _codings(r.get("code"))
    status = _concept_text(r.get("clinicalStatus")) if r.get("clinicalStatus") else "unknown"
    onset = _date(r, "onsetDateTime", "recordedDate")
    title = f"Condition: {label}"
    text = (
        f"Condition (diagnosis): {label}. Clinical status: {status}. "
        f"Onset: {onset or 'unknown'}."
    )
    return title, text, codings


def _render_medication(r: dict) -> tuple[str, str, list[Coding]]:
    concept = r.get("medicationCodeableConcept")
    label = _concept_text(concept)
    codings = _codings(concept)
    status = r.get("status", "unknown")
    authored = _date(r, "authoredOn")
    dosage = ""
    if r.get("dosageInstruction"):
        dosage = r["dosageInstruction"][0].get("text", "")
    title = f"Medication: {label}"
    text = (
        f"Medication request: {label}. Status: {status}. "
        f"Authored: {authored or 'unknown'}."
    )
    if dosage:
        text += f" Dosage: {dosage}."
    return title, text, codings


def _render_observation(r: dict) -> tuple[str, str, list[Coding]]:
    label = _concept_text(r.get("code"))
    codings = _codings(r.get("code"))
    when = _date(r, "effectiveDateTime", "issued")
    value = "no recorded value"
    if "valueQuantity" in r:
        vq = r["valueQuantity"]
        value = f"{vq.get('value')} {vq.get('unit', '')}".strip()
    elif "valueCodeableConcept" in r:
        value = _concept_text(r["valueCodeableConcept"])
    elif "valueString" in r:
        value = r["valueString"]
    title = f"Observation: {label}"
    text = f"Observation (lab/vital): {label}. Value: {value}. Date: {when or 'unknown'}."
    return title, text, codings


def _render_allergy(r: dict) -> tuple[str, str, list[Coding]]:
    label = _concept_text(r.get("code"))
    codings = _codings(r.get("code"))
    criticality = r.get("criticality", "unknown")
    title = f"Allergy/Intolerance: {label}"
    text = f"Allergy/intolerance: {label}. Criticality: {criticality}."
    return title, text, codings


def _render_encounter(r: dict) -> tuple[str, str, list[Coding]]:
    label = "Encounter"
    if r.get("type"):
        label = _concept_text(r["type"][0])
    codings = _codings(r["type"][0]) if r.get("type") else []
    when = _date(r, "period")
    reason = _concept_text(r["reasonCode"][0]) if r.get("reasonCode") else None
    title = f"Encounter: {label}"
    text = f"Encounter: {label}. Date: {when or 'unknown'}."
    if reason:
        text += f" Reason: {reason}."
    return title, text, codings


_RENDERERS = {
    "Patient": _render_patient,
    "Condition": _render_condition,
    "MedicationRequest": _render_medication,
    "Observation": _render_observation,
    "AllergyIntolerance": _render_allergy,
    "Encounter": _render_encounter,
}


def parse_bundle(bundle: dict) -> tuple[str, list[ParsedResource]]:
    """Return (patient_id, parsed resources). Assumes a single-patient bundle,
    as Synthea produces one bundle per patient."""
    entries = [e.get("resource", {}) for e in bundle.get("entry", [])]

    patient = next((r for r in entries if r.get("resourceType") == "Patient"), None)
    if patient is None:
        raise ValueError("Bundle has no Patient resource")
    patient_id = patient["id"]

    parsed: list[ParsedResource] = []
    for r in entries:
        rtype = r.get("resourceType")
        if rtype not in SUPPORTED_TYPES:
            continue
        title, txt, codings = _RENDERERS[rtype](r)
        parsed.append(
            ParsedResource(
                resource_type=rtype,
                resource_id=r["id"],
                patient_id=patient_id,
                title=title,
                text=txt,
                date=_extract_date(rtype, r),
                codings=codings,
            )
        )
    return patient_id, parsed


def _extract_date(rtype: str, r: dict) -> str | None:
    keys = {
        "Condition": ("onsetDateTime", "recordedDate"),
        "MedicationRequest": ("authoredOn",),
        "Observation": ("effectiveDateTime", "issued"),
        "Encounter": ("period",),
    }.get(rtype, ())
    return _date(r, *keys) if keys else None
