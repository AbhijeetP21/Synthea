"""Eval harness runner + CI gate.

Two layers:

* **Deterministic** (no API key — embeddings + DB only): retrieval
  precision/recall against labeled source_ids, and *abstention reachability* —
  out-of-record questions must retrieve zero evidence (which guarantees the
  service abstains) while answerable ones must retrieve something. This layer is
  the hard CI merge gate.
* **LLM** (needs a chat key; judge needs a judge key): runs the real Q&A path
  for true abstention correctness + a `must_include` content check, and the
  LLM-as-judge for groundedness / hallucination rate. Advisory by default
  (report, don't fail) — pass --strict to gate on it too.

Usage:
    mise run eval                 # full run if a key is set, else deterministic only
    uv run python -m evals.run --no-llm          # deterministic only (CI default)
    uv run python -m evals.run --strict --out evals/last_run.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from app.config import get_settings
from app.qa.service import QAService
from app.retrieval.retriever import Retriever
from evals.metrics import (
    abstention_correct,
    groundedness,
    hallucination_rate,
    mean,
    retrieval_pr,
)

GOLD_DEFAULT = Path(__file__).parent / "gold_set.yaml"

# Gate thresholds (overridable via CLI).
THRESHOLDS = {
    "retrieval_recall": 0.80,        # deterministic, gated
    "abstention_reachability": 0.85,  # deterministic, gated
    "abstention_accuracy": 0.85,     # llm, advisory unless --strict
    "groundedness": 0.90,            # llm, advisory unless --strict
    "hallucination_rate": 0.10,      # llm (max), advisory unless --strict
    "must_include": 0.90,            # llm, advisory unless --strict
}


def load_gold(path: Path) -> tuple[str, list[dict]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["patient_id"], data["items"]


def run(gold_path: Path, *, run_llm: bool, run_judge: bool) -> dict:
    patient_id, items = load_gold(gold_path)
    retriever = Retriever()
    service = QAService() if run_llm else None
    judge = None
    if run_llm and run_judge:
        from evals.judge import Judge

        judge = Judge()

    recalls: list[float] = []
    precisions: list[float] = []
    reach_hits = 0
    reach_total = 0
    abst_hits = 0
    abst_total = 0
    must_hits = 0
    must_total = 0
    sup_total = 0
    sent_total = 0
    rows: list[dict] = []

    for item in items:
        q = item["question"]
        should_abstain = item["should_abstain"]
        retrieved = [s.source_id for s in retriever.retrieve(patient_id, q)]
        relevant = item.get("relevant_source_ids", [])

        row: dict = {"id": item["id"], "should_abstain": should_abstain,
                     "retrieved": len(retrieved)}

        if relevant:
            pr = retrieval_pr(retrieved, relevant)
            recalls.append(pr.recall)
            precisions.append(pr.precision)
            row["recall"] = round(pr.recall, 3)
            row["precision"] = round(pr.precision, 3)

        # Abstention reachability (key-free): only assert what the relevance gate
        # alone can guarantee — answerable items must retrieve something, and
        # off-topic items must retrieve nothing. Clinically-adjacent
        # (out_of_record) abstentions depend on the grounded LLM layer and are
        # measured by abstention_accuracy below, not here.
        kind = item.get("abstain_kind")
        if not should_abstain:
            reach_total += 1
            reachable = len(retrieved) > 0
            reach_hits += int(reachable)
            row["reachable"] = reachable
        elif kind == "off_topic":
            reach_total += 1
            reachable = len(retrieved) == 0
            reach_hits += int(reachable)
            row["reachable"] = reachable

        if service is not None:
            resp = service.ask(patient_id, q)
            abst_total += 1
            ok = abstention_correct(resp.abstained, should_abstain)
            abst_hits += int(ok)
            row["abstained"] = resp.abstained
            row["abstention_ok"] = ok

            if not resp.abstained:
                answer_text = " ".join(s.text for s in resp.sentences).lower()
                for term in item.get("must_include", []):
                    must_total += 1
                    must_hits += int(term.lower() in answer_text)
                if judge is not None:
                    j = judge.judge_answer(resp)
                    sup_total += j.supported
                    sent_total += j.total
                    row["supported"] = f"{j.supported}/{j.total}"
        rows.append(row)

    metrics = {
        "retrieval_recall": round(mean(recalls), 3),
        "retrieval_precision": round(mean(precisions), 3),
        "abstention_reachability": round(reach_hits / reach_total, 3) if reach_total else None,
    }
    if service is not None:
        metrics["abstention_accuracy"] = round(abst_hits / abst_total, 3) if abst_total else None
        metrics["must_include"] = round(must_hits / must_total, 3) if must_total else None
    if judge is not None:
        metrics["groundedness"] = round(groundedness(sup_total, sent_total), 3)
        metrics["hallucination_rate"] = round(hallucination_rate(sup_total, sent_total), 3)

    return {
        "patient_id": patient_id,
        "n_items": len(items),
        "ran_llm": service is not None,
        "ran_judge": judge is not None,
        "metrics": metrics,
        "rows": rows,
    }


def gate(metrics: dict, *, strict: bool) -> list[str]:
    """Return a list of threshold failures. Deterministic metrics always gate;
    LLM metrics gate only with --strict."""
    failures = []
    det = {"retrieval_recall", "abstention_reachability"}
    higher_is_better = {
        "retrieval_recall", "abstention_reachability", "abstention_accuracy",
        "groundedness", "must_include",
    }
    for key, limit in THRESHOLDS.items():
        val = metrics.get(key)
        if val is None:
            continue
        if key not in det and not strict:
            continue
        ok = val >= limit if key in higher_is_better else val <= limit
        if not ok:
            cmp = ">=" if key in higher_is_better else "<="
            failures.append(f"{key}={val} (need {cmp} {limit})")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description="Run the clinical-RAG eval harness.")
    p.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    p.add_argument("--no-llm", action="store_true",
                   help="deterministic metrics only (no API key needed)")
    p.add_argument("--no-judge", action="store_true",
                   help="run the Q&A path but skip the LLM-as-judge")
    p.add_argument("--strict", action="store_true",
                   help="also gate on LLM metrics (default: advisory)")
    p.add_argument("--out", type=Path, help="write the full result JSON here")
    args = p.parse_args()

    settings = get_settings()
    has_key = bool(settings.chat_api_key)
    run_llm = not args.no_llm and has_key
    run_judge = run_llm and not args.no_judge

    if not args.no_llm and not has_key:
        print("! No CHAT_API_KEY set — running deterministic metrics only.\n")

    result = run(args.gold, run_llm=run_llm, run_judge=run_judge)

    # Scorecard
    print(f"Eval over {result['n_items']} gold items "
          f"(llm={'on' if run_llm else 'off'}, judge={'on' if run_judge else 'off'})\n")
    for key, val in result["metrics"].items():
        if val is not None:
            print(f"  {key:26s} {val}")
    print()

    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}\n")

    failures = gate(result["metrics"], strict=args.strict)
    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
