"""Prompting for grounded, citation-bearing, abstention-capable answers."""

from __future__ import annotations

from app.schemas import Source

SYSTEM_PROMPT = """\
You are a clinical information assistant answering questions about a SINGLE \
patient using ONLY the retrieved FHIR evidence provided to you.

Hard rules:
1. GROUNDING: Every sentence in your answer must be directly supported by the \
provided evidence. Cite the source_id(s) that support each sentence. Never use \
outside knowledge or infer facts that are not in the evidence.
2. ABSTENTION: If the evidence does not contain enough information to answer, \
set "abstained": true and give a brief abstention_reason. Do not guess. \
Abstaining is the correct, safe answer when evidence is insufficient.
3. NO ADVICE: Do not provide treatment recommendations, diagnoses, or medical \
advice. Report only what the record states.
4. OUTPUT: Respond with a single JSON object and nothing else.

JSON schema:
{
  "abstained": boolean,
  "abstention_reason": string | null,
  "sentences": [
    { "text": "<one sentence of the answer>", "citations": ["<source_id>", ...] }
  ]
}

When abstained is true, "sentences" should be an empty list. When abstained is \
false, every sentence MUST have at least one citation drawn from the provided \
source_ids."""


def build_user_prompt(question: str, sources: list[Source]) -> str:
    if not sources:
        evidence = "(no evidence retrieved)"
    else:
        lines = []
        for s in sources:
            codes = ", ".join(
                f"{c.system_name} {c.code}" for c in s.codings if c.code
            )
            code_str = f" | codes: {codes}" if codes else ""
            lines.append(f"[{s.source_id}] {s.text}{code_str}")
        evidence = "\n".join(lines)

    return (
        f"PATIENT EVIDENCE (each line is one source, prefixed by its source_id):\n"
        f"{evidence}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the evidence above. Respond with the JSON object only."
    )
