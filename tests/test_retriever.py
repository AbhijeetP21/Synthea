"""The relevance gate is the Phase 2 abstention mechanism: chunks beyond the
distance ceiling are dropped, so off-topic questions retrieve nothing and the
Q&A service abstains. Tests mock the embedder and DB to stay offline."""

import app.retrieval.retriever as retriever_mod
from app.retrieval.retriever import Retriever
from app.schemas import Source


class _FakeEmbedder:
    def embed_query(self, query):
        return [0.0] * 4


def _src(source_id):
    rt, rid = source_id.split("/", 1)
    return Source(
        source_id=source_id, resource_type=rt, resource_id=rid,
        patient_id="pat-1", title=source_id, text=source_id,
    )


def _build_retriever(monkeypatch, scored):
    monkeypatch.setattr(retriever_mod, "get_embeddings_client", lambda: _FakeEmbedder())
    monkeypatch.setattr(retriever_mod, "semantic_search", lambda *a, **k: scored)
    r = Retriever()
    r._max_distance = 0.40  # pin to the calibrated default regardless of env
    return r


def test_keeps_only_sources_within_distance_ceiling(monkeypatch):
    scored = [
        (_src("MedicationRequest/m1"), 0.27),  # relevant
        (_src("Condition/c1"), 0.34),          # relevant, near the ceiling
        (_src("Observation/o1"), 0.41),        # just over -> dropped
        (_src("Encounter/e1"), 0.55),          # far -> dropped
    ]
    r = _build_retriever(monkeypatch, scored)
    kept = r.retrieve("pat-1", "what medications?")
    assert [s.source_id for s in kept] == ["MedicationRequest/m1", "Condition/c1"]


def test_off_topic_query_retrieves_nothing(monkeypatch):
    # Every chunk is beyond the ceiling (mirrors measured off-topic distances).
    scored = [(_src("Condition/c1"), 0.45), (_src("Observation/o1"), 0.56)]
    r = _build_retriever(monkeypatch, scored)
    assert r.retrieve("pat-1", "what is the capital of France?") == []
