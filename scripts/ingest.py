"""CLI to ingest a Synthea FHIR bundle into the vector store.

Usage:
    mise run ingest data/sample/patient_bundle.json
    uv run python -m scripts.ingest data/sample/patient_bundle.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.ingest.pipeline import ingest_bundle


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.ingest <path-to-fhir-bundle.json>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"error: file not found: {path}")
        return 1

    print(f"Ingesting {path} ...")
    summary = ingest_bundle(path)
    print(f"  patient_id : {summary['patient_id']}")
    print(f"  ingested   : {summary['ingested']} chunks")
    for rtype, n in sorted(summary.get("by_type", {}).items()):
        print(f"    - {rtype}: {n}")

    phi = summary.get("phi", {})
    detected, redacted = phi.get("detected", {}), phi.get("redacted", {})
    if detected:
        print("  PHI detected (HIPAA Safe Harbor identifiers):")
        for ent, n in sorted(detected.items(), key=lambda kv: (-kv[1], kv[0])):
            mark = f"redacted x{redacted[ent]}" if ent in redacted else "flagged, kept"
            print(f"    - {ent}: {n} ({mark})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
