"""Dashboard rendering helpers (Phase 5) — pure, no Streamlit."""

from app.schemas import AnswerSentence, AskResponse, Coding, Source
from app.web.render import (
    answer_to_html,
    format_codes,
    number_citations,
    ordered_sources,
    sentence_to_html,
)


def _resp() -> AskResponse:
    return AskResponse(
        patient_id="p1",
        question="meds?",
        abstained=False,
        sentences=[
            AnswerSentence(text="On amlodipine.", citations=["MedicationRequest/m1"]),
            AnswerSentence(
                text="Also insulin.",
                citations=["MedicationRequest/m2", "MedicationRequest/m1"],
            ),
        ],
        sources=[
            Source(source_id="MedicationRequest/m1", resource_type="MedicationRequest",
                   resource_id="m1", patient_id="p1", title="Medication: amlodipine",
                   text="amLODIPine 2.5 MG",
                   codings=[Coding(system_name="RxNorm", code="308136")]),
            Source(source_id="MedicationRequest/m2", resource_type="MedicationRequest",
                   resource_id="m2", patient_id="p1", title="Medication: insulin",
                   text="Humulin"),
        ],
    )


def test_number_citations_stable_by_first_appearance():
    r = _resp()
    numbering = number_citations(r.sentences)
    assert numbering == {"MedicationRequest/m1": 1, "MedicationRequest/m2": 2}


def test_sentence_html_has_sorted_unique_badges():
    r = _resp()
    numbering = number_citations(r.sentences)
    # Second sentence cites m2 and m1 -> badges sorted [1][2], no dupes.
    html = sentence_to_html(r.sentences[1], numbering)
    assert "[1]" in html and "[2]" in html
    assert html.index("[1]") < html.index("[2]")
    assert 'href="#src-1"' in html


def test_html_escapes_redaction_tokens():
    # PHI tokens like <PERSON> must not break the HTML render.
    s = AnswerSentence(text="Patient <PERSON> on drug.", citations=[])
    html = sentence_to_html(s, {})
    assert "&lt;PERSON&gt;" in html
    assert "<PERSON>" not in html


def test_ordered_sources_follow_citation_numbers():
    r = _resp()
    numbering = number_citations(r.sentences)
    ordered = ordered_sources(r, numbering)
    assert [n for n, _ in ordered] == [1, 2]
    assert ordered[0][1].source_id == "MedicationRequest/m1"


def test_answer_html_and_codes():
    r = _resp()
    numbering = number_citations(r.sentences)
    assert "On amlodipine." in answer_to_html(r, numbering)
    assert format_codes(r.sources[0]) == "RxNorm 308136"
    assert format_codes(r.sources[1]) == ""
