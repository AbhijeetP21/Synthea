"""Eval metric functions — pure and deterministic."""

from evals.metrics import (
    abstention_correct,
    groundedness,
    hallucination_rate,
    mean,
    retrieval_pr,
)


def test_retrieval_pr_perfect():
    pr = retrieval_pr(["A/1", "B/2"], ["A/1", "B/2"])
    assert pr.precision == 1.0 and pr.recall == 1.0


def test_retrieval_pr_partial():
    # retrieved 4, of which 1 is relevant; 2 relevant total -> P=0.25, R=0.5
    pr = retrieval_pr(["A/1", "x", "y", "z"], ["A/1", "B/2"])
    assert pr.precision == 0.25
    assert pr.recall == 0.5


def test_retrieval_pr_recall_caps_at_one():
    pr = retrieval_pr(["A/1", "A/1dup-not-real", "B/2"], ["A/1"])
    assert pr.recall == 1.0


def test_abstention_correct():
    assert abstention_correct(True, True)
    assert abstention_correct(False, False)
    assert not abstention_correct(True, False)
    assert not abstention_correct(False, True)


def test_groundedness_and_hallucination_are_complementary():
    assert groundedness(8, 10) == 0.8
    assert hallucination_rate(8, 10) == 0.2
    # empty answer: nothing unsupported.
    assert groundedness(0, 0) == 1.0
    assert hallucination_rate(0, 0) == 0.0


def test_mean_empty():
    assert mean([]) == 0.0
    assert mean([0.5, 1.0]) == 0.75
