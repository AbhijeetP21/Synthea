"""Pure rendering helpers for the dashboard — no Streamlit, so they're testable.

The job here is turning an AskResponse into inline-cited HTML: every sentence
keeps superscript [n] markers that point at numbered source cards, which is the
visible-traceability requirement from the brief.
"""

from __future__ import annotations

from app.schemas import AnswerSentence, AskResponse, Source


def number_citations(sentences: list[AnswerSentence]) -> dict[str, int]:
    """Map each cited source_id to a stable [1..] number, by first appearance."""
    order: list[str] = []
    for sent in sentences:
        for cid in sent.citations:
            if cid not in order:
                order.append(cid)
    return {cid: i + 1 for i, cid in enumerate(order)}


def sentence_to_html(sentence: AnswerSentence, numbering: dict[str, int]) -> str:
    """Render one sentence with superscript citation links to its source cards."""
    nums = sorted({numbering[c] for c in sentence.citations if c in numbering})
    badges = "".join(
        f'<sup><a href="#src-{n}" title="source {n}">[{n}]</a></sup>' for n in nums
    )
    sep = " " if badges else ""
    return f"{_escape(sentence.text)}{sep}{badges}"


def answer_to_html(response: AskResponse, numbering: dict[str, int]) -> str:
    """Render the full grounded answer as one flowing paragraph with inline cites."""
    return " ".join(sentence_to_html(s, numbering) for s in response.sentences)


def format_codes(source: Source) -> str:
    return ", ".join(f"{c.system_name} {c.code}" for c in source.codings if c.code)


def ordered_sources(response: AskResponse, numbering: dict[str, int]) -> list[tuple[int, Source]]:
    """Cited sources paired with their citation number, in citation order.
    Any cited id missing from response.sources is skipped (shouldn't happen — the
    service only returns resolved citations)."""
    by_id = {s.source_id: s for s in response.sources}
    pairs = [(n, by_id[cid]) for cid, n in numbering.items() if cid in by_id]
    return sorted(pairs, key=lambda p: p[0])


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
