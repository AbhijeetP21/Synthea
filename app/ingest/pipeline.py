"""Ingestion orchestrator: parse -> redact (PHI) -> embed -> store.

The order matters: redaction happens before embedding and before anything is
persisted, so no un-redacted text ever reaches the model or the vector store.
"""

from __future__ import annotations

from pathlib import Path

from app.db import Chunk, init_db, upsert_chunks
from app.ingest.fhir_parser import ParsedResource, load_bundle, parse_bundle
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

    texts = [redactor.redact(_embeddable_text(r)) for r in resources]
    vectors = embedder.embed_documents(texts)

    rows = [
        Chunk(
            source_id=r.source_id,
            resource_type=r.resource_type,
            resource_id=r.resource_id,
            patient_id=r.patient_id,
            title=r.title,
            body=body,
            date=r.date,
            codings=[c.model_dump() for c in r.codings],
            embedding=vec,
        )
        for r, body, vec in zip(resources, texts, vectors, strict=True)
    ]
    n = upsert_chunks(rows)

    counts: dict[str, int] = {}
    for r in resources:
        counts[r.resource_type] = counts.get(r.resource_type, 0) + 1
    return {"patient_id": patient_id, "ingested": n, "by_type": counts}
