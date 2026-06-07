import json

from app.qa.service import QAService, _parse_llm_json
from app.schemas import Source


class FakeRetriever:
    def __init__(self, sources):
        self._sources = sources

    def retrieve(self, patient_id, query, k=None):
        return self._sources


class FakeChat:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, *, system, user):
        return self._payload


def _src(source_id, rtype, text):
    rt, rid = source_id.split("/", 1)
    return Source(
        source_id=source_id,
        resource_type=rtype,
        resource_id=rid,
        patient_id="pat-1",
        title=text,
        text=text,
    )


SOURCES = [
    _src("MedicationRequest/med-1", "MedicationRequest", "Lisinopril 10 MG"),
    _src("AllergyIntolerance/alg-1", "AllergyIntolerance", "Penicillin allergy"),
]


def test_abstains_when_no_evidence_without_calling_model():
    svc = QAService(retriever=FakeRetriever([]), chat=FakeChat("should not be used"))
    resp = svc.ask("pat-1", "What meds?")
    assert resp.abstained is True
    assert resp.sentences == []


def test_grounded_answer_keeps_only_valid_citations():
    payload = json.dumps(
        {
            "abstained": False,
            "sentences": [
                {
                    "text": "The patient takes Lisinopril 10 MG.",
                    "citations": ["MedicationRequest/med-1", "Bogus/x-1"],
                }
            ],
        }
    )
    svc = QAService(retriever=FakeRetriever(SOURCES), chat=FakeChat(payload))
    resp = svc.ask("pat-1", "What medications?")
    assert resp.abstained is False
    assert len(resp.sentences) == 1
    # The hallucinated citation is dropped; only the resolvable one remains.
    assert resp.sentences[0].citations == ["MedicationRequest/med-1"]
    assert [s.source_id for s in resp.sources] == ["MedicationRequest/med-1"]


def test_falls_back_to_abstention_when_no_citation_resolves():
    payload = json.dumps(
        {
            "abstained": False,
            "sentences": [{"text": "Unsupported claim.", "citations": ["Bogus/x-1"]}],
        }
    )
    svc = QAService(retriever=FakeRetriever(SOURCES), chat=FakeChat(payload))
    resp = svc.ask("pat-1", "anything?")
    assert resp.abstained is True
    assert "grounded" in (resp.abstention_reason or "").lower()


def test_respects_explicit_model_abstention():
    payload = json.dumps(
        {"abstained": True, "abstention_reason": "Not in record.", "sentences": []}
    )
    svc = QAService(retriever=FakeRetriever(SOURCES), chat=FakeChat(payload))
    resp = svc.ask("pat-1", "What is the blood type?")
    assert resp.abstained is True
    assert resp.abstention_reason == "Not in record."


def test_parse_llm_json_handles_fenced_and_prose():
    fenced = """```json
    {"abstained": false, "sentences": [{"text": "x", "citations": ["A/1"]}]}
    ```"""
    parsed = _parse_llm_json(fenced)
    assert parsed.abstained is False
    assert parsed.sentences[0].citations == ["A/1"]

    prose = 'Here is the answer: {"abstained": true, "abstention_reason": "no", "sentences": []} Thanks!'
    parsed2 = _parse_llm_json(prose)
    assert parsed2.abstained is True


def test_parse_llm_json_unparseable_defaults_to_abstention():
    parsed = _parse_llm_json("the model rambled with no json")
    assert parsed.abstained is True
