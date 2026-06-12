"""Live smoke test: exercise the full grounded-Q&A path against the configured chat provider."""

import json
import sys

from app.qa.service import QAService

PATIENT_ID = "38c50ee6-b356-4a70-2fd9-44e304a734cd"


def run(question: str) -> None:
    svc = QAService()
    resp = svc.ask(PATIENT_ID, question)
    print("=" * 80)
    print("Q:", question)
    print("abstained:", resp.abstained)
    if resp.abstention_reason:
        print("reason:", resp.abstention_reason)
    for s in resp.sentences:
        print(f"  - {s.text}")
        print(f"      cites: {s.citations}")
    print(f"sources resolved: {len(resp.sources)}")
    for src in resp.sources:
        print(f"      [{src.source_id}] {src.title}")


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "What medications is this patient on, and are there any documented allergies?",
        "Does this patient have a recorded diagnosis of diabetes?",
    ]
    for q in questions:
        run(q)
