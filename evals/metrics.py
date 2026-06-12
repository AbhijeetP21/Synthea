"""Pure metric functions — deterministic, no I/O, easy to unit test.

These cover the four pillars from the brief: groundedness, hallucination rate,
retrieval precision/recall, and abstention correctness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PR:
    precision: float
    recall: float


def retrieval_pr(retrieved: list[str], relevant: list[str]) -> PR:
    """Precision/recall of retrieved source_ids against the labeled relevant set."""
    r = set(relevant)
    g = set(retrieved)
    if not r:
        # No labeled relevant set: undefined, caller should skip.
        return PR(precision=1.0 if not g else 0.0, recall=1.0)
    hits = len(r & g)
    precision = hits / len(g) if g else 0.0
    recall = hits / len(r)
    return PR(precision=precision, recall=recall)


def abstention_correct(abstained: bool, should_abstain: bool) -> bool:
    return abstained == should_abstain


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def groundedness(total_supported: int, total_sentences: int) -> float:
    """Fraction of answer sentences supported by their cited evidence."""
    return total_supported / total_sentences if total_sentences else 1.0


def hallucination_rate(total_supported: int, total_sentences: int) -> float:
    """Fraction of answer sentences NOT supported by their cited evidence."""
    if not total_sentences:
        return 0.0
    return (total_sentences - total_supported) / total_sentences
