"""Ingestion orchestrator: parse -> redact (PHI) -> embed -> store.

The order matters: redaction happens before embedding and before anything is
persisted, so no un-redacted text ever reaches the model or the vector store.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.db import Chunk, init_db, upsert_chunks
from app.ingest.fhir_parser import (
    ParsedResource,
    collect_phi_hints,
    load_bundle,
    parse_bundle,
)
from app.ingest.phi import get_redactor
from app.llm import get_embeddings_client


def _embeddable_text(r: ParsedResource) -> str:
    """Compose the text we embed and show the model. Codes are appended so exact
    clinical terms (drug names, code values) are present for retrieval."""
    parts = [r.text]
    for c in r.codings:
        if c.code:
            display = f" {c.display}" if c.display else ""
            parts.append(f"[{c.system_name or 'code'} {c.code}{display}]")
    return " ".join(parts)


def ingest_bundle(path: str | Path) -> dict:
    """Ingest one Synthea bundle. Returns a small summary dict."""
    init_db()
    redactor = get_redactor()
    embedder = get_embeddings_client()

    bundle = load_bundle(path)
    patient_id, resources = parse_bundle(bundle)
    if not resources:
        return {"patient_id": patient_id, "ingested": 0}

    # PHI stage: redact title + body before anything is embedded or stored.
    # Structured identifiers (name, address) seed the deny-list so detection does
    # not rely on NER alone. Detections/redactions are tallied for the report.
    hints = collect_phi_hints(bundle)
    detected: Counter[str] = Counter()
    redacted: Counter[str] = Counter()
    titles: list[str] = []
    bodies: list[str] = []
    for r in resources:
        t = redactor.redact(r.title, hints=hints)
        b = redactor.redact(_embeddable_text(r), hints=hints)
        titles.append(t.text)
        bodies.append(b.text)
        for res in (t, b):
            detected.update(res.detected)
            redacted.update(res.redacted)

    vectors = embedder.embed_documents(bodies)

    rows = [
        Chunk(
            source_id=r.source_id,
            resource_type=r.resource_type,
            resource_id=r.resource_id,
            patient_id=r.patient_id,
            title=title,
            body=body,
            date=r.date,
            codings=[c.model_dump() for c in r.codings],
            embedding=vec,
        )
        for r, title, body, vec in zip(resources, titles, bodies, vectors, strict=True)
    ]
    n = upsert_chunks(rows)

    counts: dict[str, int] = {}
    for r in resources:
        counts[r.resource_type] = counts.get(r.resource_type, 0) + 1
    return {
        "patient_id": patient_id,
        "ingested": n,
        "by_type": counts,
        "phi": {"detected": dict(detected), "redacted": dict(redacted)},
    }
