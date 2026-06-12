"""LLM-as-judge for groundedness / hallucination.

For each answer sentence we ask the judge whether the sentence is *fully
supported by only the evidence cited for it*. This is the metric that catches a
model asserting something the cited FHIR resource does not actually say. The
judge runs through its own provider seam (`get_judge_client`) so it can be a
different model than the one under test.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from app.llm import ChatClient, get_judge_client
from app.schemas import AskResponse

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYSTEM = """\
You are a strict evaluation judge for a clinical question-answering system. \
You are NOT answering the question. For each numbered CLAIM you are given the \
exact evidence text that the system cited for it. Decide whether the claim is \
FULLY supported by ONLY that cited evidence — no outside knowledge, no \
plausible inference beyond what the text states.

Respond with a single JSON object and nothing else:
{"verdicts": [{"i": <claim number>, "supported": <true|false>, "reason": "<short>"}]}"""


class SentenceVerdict(BaseModel):
    i: int
    supported: bool
    reason: str = ""


class AnswerJudgement(BaseModel):
    total: int
    supported: int

    @property
    def unsupported(self) -> int:
        return self.total - self.supported


def _build_prompt(response: AskResponse) -> str:
    by_id = {s.source_id: s for s in response.sources}
    lines = [f"QUESTION: {response.question}", "", "CLAIMS:"]
    for idx, sent in enumerate(response.sentences):
        lines.append(f"[{idx}] {sent.text}")
        for cid in sent.citations:
            src = by_id.get(cid)
            evidence = src.text if src else "(cited source not found)"
            lines.append(f"      cited {cid}: {evidence}")
    lines.append("")
    lines.append("Return the JSON verdict object now.")
    return "\n".join(lines)


def _parse(raw: str, n: int) -> list[SentenceVerdict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    match = _JSON_RE.search(text)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
        verdicts = [SentenceVerdict(**v) for v in data.get("verdicts", [])]
    except Exception:
        # Unparseable judge output -> treat every claim as unsupported (strict).
        return [SentenceVerdict(i=i, supported=False, reason="unparseable judge output")
                for i in range(n)]
    seen = {v.i for v in verdicts}
    verdicts.extend(
        SentenceVerdict(i=i, supported=False, reason="missing verdict") for i in range(n)
        if i not in seen
    )
    return verdicts


class Judge:
    def __init__(self, client: ChatClient | None = None) -> None:
        self._client = client or get_judge_client()

    def judge_answer(self, response: AskResponse) -> AnswerJudgement:
        """Score one grounded (non-abstained) answer. Empty answer -> 0/0."""
        n = len(response.sentences)
        if n == 0:
            return AnswerJudgement(total=0, supported=0)
        raw = self._client.complete(system=JUDGE_SYSTEM, user=_build_prompt(response))
        verdicts = _parse(raw, n)
        supported = sum(1 for v in verdicts if v.supported and 0 <= v.i < n)
        return AnswerJudgement(total=n, supported=min(supported, n))
