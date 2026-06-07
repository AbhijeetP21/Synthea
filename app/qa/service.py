"""Q&A service: retrieve -> prompt -> structured answer -> enforce grounding.

The grounding gate is the heart of the safety story: any sentence that does not
carry a citation resolvable to a retrieved source is dropped, and if nothing
grounded survives, the system abstains rather than asserting an unsupported
clinical claim.
"""

from __future__ import annotations

import json
import re

from app.llm import get_chat_client
from app.qa.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import Retriever
from app.schemas import AnswerSentence, AskResponse, LLMAnswer, Source

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> LLMAnswer:
    """Tolerant extraction: handles bare JSON, ```json fences, and trailing prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return LLMAnswer.model_validate_json(text)
    except Exception:
        pass
    match = _JSON_RE.search(text)
    if match:
        try:
            return LLMAnswer.model_validate(json.loads(match.group(0)))
        except Exception:
            pass
    # Could not parse a structured answer -> safest outcome is abstention.
    return LLMAnswer(
        abstained=True,
        abstention_reason="The model did not return a parseable grounded answer.",
    )


class QAService:
    def __init__(self, retriever: Retriever | None = None, chat=None) -> None:
        # Dependencies are injectable for testing; built lazily otherwise so the
        # heavy embedding model / DB are only touched when actually serving.
        self._retriever = retriever or Retriever()
        self._chat = chat or get_chat_client()

    def ask(self, patient_id: str, question: str) -> AskResponse:
        sources = self._retriever.retrieve(patient_id, question)
        by_id: dict[str, Source] = {s.source_id: s for s in sources}

        # No evidence at all -> abstain without calling the model.
        if not sources:
            return AskResponse(
                patient_id=patient_id,
                question=question,
                abstained=True,
                abstention_reason="No matching information was found in this patient's record.",
            )

        raw = self._chat.complete(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(question, sources),
        )
        answer = _parse_llm_json(raw)

        if answer.abstained:
            return AskResponse(
                patient_id=patient_id,
                question=question,
                abstained=True,
                abstention_reason=answer.abstention_reason
                or "The retrieved evidence was insufficient to answer.",
            )

        # Grounding gate: keep only citations that resolve to retrieved sources,
        # and only sentences that retain at least one valid citation.
        grounded: list[AnswerSentence] = []
        cited_ids: set[str] = set()
        for sent in answer.sentences:
            valid = [cid for cid in sent.citations if cid in by_id]
            if not valid:
                continue
            grounded.append(AnswerSentence(text=sent.text, citations=valid))
            cited_ids.update(valid)

        if not grounded:
            return AskResponse(
                patient_id=patient_id,
                question=question,
                abstained=True,
                abstention_reason="The answer could not be grounded in the retrieved evidence.",
            )

        return AskResponse(
            patient_id=patient_id,
            question=question,
            abstained=False,
            sentences=grounded,
            sources=[by_id[cid] for cid in by_id if cid in cited_ids],
        )
