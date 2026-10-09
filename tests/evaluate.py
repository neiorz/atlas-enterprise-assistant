"""DeepEval evaluation script (FR-J).

Reads the 10 gold cases from tests/eval_cases.jsonl, runs each one through the
full LangGraph workflow, scores it with the five required DeepEval metrics plus
a custom routing-accuracy check, and writes outputs/eval_report.json.

Usage:
    python tests/evaluate.py                 # full run
    python tests/evaluate.py --limit 2       # smoke test on first N cases
    python tests/evaluate.py --skip-llm      # routing accuracy only (no judge calls)

Design notes:
- Each case runs in a FRESH session (new session_id), so session memory from
  case N can never leak into case N+1 and inflate the scores.
- `context` (the gold standard the answer must NOT contradict) is the full text
  of the documents listed in `expected_sources`; `retrieval_context` is what our
  pipeline actually retrieved. Hallucination/Precision/Recall compare the two.
- Routing accuracy uses the router's raw choice from run_logs.jsonl (the
  `router_decision` event), so a low-confidence fallback to "multi" is visible
  in the report instead of silently counted as a wrong domain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow `python tests/evaluate.py` from the repo root (FR-K1: five-command quickstart).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from src.loaders import load_document, strip_markdown_headings
from src.settings import settings

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "contextual_precision",
    "contextual_recall",
    "hallucination",
)

METRIC_CLASSES = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "hallucination": HallucinationMetric,
}


def _api_key_ready() -> bool:
    """True only when a real key is configured — catches the untouched
    .env.example placeholder so users get a fix-it message, not a 400."""
    key = settings.google_api_key.strip()
    return bool(key) and not key.startswith("your-")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_eval_cases() -> list[dict]:
    """Read the gold cases, failing loudly if the bundle is missing (FR-J1)."""
    path: Path = settings.eval_cases_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — unzip atlas-corpus.zip so that "
            f"tests/eval_cases.jsonl sits at the repo root (see README Quick Start)."
        )
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_gold_context(filenames: list[str]) -> list[str]:
    """Full text of the documents the answer is expected to rely on.

    These become the LLMTestCase `context` — the ground truth the judge
    compares `actual_output` against for hallucination, and the retrieval
    target for contextual recall (FR-J2).
    """
    texts: list[str] = []
    for name in filenames:
        for domain in settings.domains:
            path = settings.docs_dir / domain / name
            if path.exists():
                raw = load_document(path)
                if path.suffix.lower() == ".md":
                    raw = strip_markdown_headings(raw)
                texts.append(f"[{name}]\n{raw}")
                break
        else:
            # FR-G2: a gold source that doesn't exist means the corpus wasn't
            # extracted correctly — surface it instead of scoring against nothing.
            raise FileNotFoundError(
                f"Gold source '{name}' not found under {settings.docs_dir}/ "
                f"for any of {settings.domains}."
            )
    return texts


def find_router_decision(run_id: str) -> dict | None:
    """Re-read this run's router event from run_logs.jsonl (FR-I2, FR-I3).

    Returns the raw router choice + confidence + fallback flag, or None if the
    log line can't be found (e.g. log rotated mid-run).
    """
    path: Path = settings.run_logs_path
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a corrupt line must not kill a 10-case evaluation
            if (
                record.get("run_id") == run_id
                and record.get("event") == "router_decision"
            ):
                return record
    return None


# --------------------------------------------------------------------------
# Running cases through the workflow
# --------------------------------------------------------------------------


def run_case(case: dict) -> dict:
    """Run one gold question through the graph in a fresh session.

    Args:
        case: A single line of eval_cases.jsonl.

    Returns:
        A dict with the workflow outputs (answer, sources, chunks) plus the
        router decision recovered from the run log.
    """
    import uuid

    from src.graph import run_turn

    session_id = str(uuid.uuid4())  # fresh session per case — no memory bleed
    started = time.perf_counter()
    state = run_turn(case["input"], session_id)
    elapsed = time.perf_counter() - started

    router_event = find_router_decision(state.get("run_id", "")) or {}
    return {
        "actual_output": state.get("answer", ""),
        "sources": state.get("sources", []),
        "retrieval_context": [
            chunk.get("text", "")
            for chunk in state.get("retrieved_chunks", [])
            if chunk.get("text")
        ],
        "router_domain": router_event.get("chosen_domain", state.get("domain")),
        "router_confidence": router_event.get(
            "confidence", state.get("router_confidence")
        ),
        "fallback_applied": router_event.get("fallback_applied", False),
        "run_id": state.get("run_id"),
        "elapsed_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def build_metrics() -> dict:
    """Instantiate the five required metrics once and reuse them (FR-J3).

    Threshold 0.5 == DeepEval's conventional pass bar; strict mode off so the
    report carries continuous scores (more useful for the final report's
    per-metric commentary than a bare pass/fail).
    """
    from src.judge import get_judge

    judge = get_judge()
    metrics = {}
    for name, cls in METRIC_CLASSES.items():
        # async_mode=False keeps scoring on the synchronous path: evaluate.py
        # is a plain script with no running event loop, and the async variant
        # would spin up its own loop per metric for no benefit.
        metrics[name] = cls(
            threshold=0.5,
            model=judge,
            include_reason=True,
            strict_mode=False,
            async_mode=False,
        )
    return metrics


def score_case(metrics: dict, case: dict, result: dict) -> dict:
    """Score one case with all five metrics, capturing score/pass/reason."""
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=result["retrieval_context"] or ["(nothing retrieved)"],
        context=load_gold_context(case["expected_sources"]),
    )

    scores: dict[str, dict] = {}
    for name, metric in metrics.items():
        metric.measure(test_case)
        scores[name] = {
            "score": round(float(metric.score), 4),
            "pass": bool(metric.success),
            "reason": getattr(metric, "reason", "") or "",
        }
    return scores


def routing_check(case: dict, result: dict) -> dict:
    """Custom routing-accuracy check (FR-J4)."""
    expected = case["expected_domain"]
    predicted = result.get("router_domain")
    # A low-confidence fallback ("multi") is NOT the right domain — it means the
    # router declined to pick, so it fails accuracy but stays honest in the report.
    passed = predicted == expected
    return {
        "predicted_domain": predicted,
        "expected_domain": expected,
        "confidence": result.get("router_confidence"),
        "fallback_applied": result.get("fallback_applied", False),
        "pass": passed,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def summarize(cases_detail: list[dict]) -> dict:
    """Per-metric averages, pass/fail counts, and overall routing accuracy."""
    metrics_summary = {}
    for name in METRIC_NAMES:
        values = [
            c["scores"][name] for c in cases_detail if name in c.get("scores", {})
        ]
        if not values:
            continue
        passed = sum(1 for v in values if v["pass"])
        metrics_summary[name] = {
            "average_score": round(sum(v["score"] for v in values) / len(values), 4),
            "pass_count": passed,
            "fail_count": len(values) - passed,
            "threshold": 0.5,
        }

    routing = [c["routing"] for c in cases_detail if "routing" in c]
    routing_pass = sum(1 for r in routing if r["pass"])
    return {
        "metrics": metrics_summary,
        "routing_accuracy": {
            "overall": round(routing_pass / len(routing), 4) if routing else 0.0,
            "correct": routing_pass,
            "total": len(routing),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepEval evaluation suite.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only run the first N cases."
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the five LLM-judge metrics; report routing accuracy only.",
    )
    args = parser.parse_args()

    if not _api_key_ready():
        print(
            "ERROR: GOOGLE_API_KEY is not set (or is still the placeholder).\n"
            "  1. Get a free key: https://aistudio.google.com/apikey\n"
            "  2. Put it in .env as:  GOOGLE_API_KEY=AIza...\n"
            "See .env.example for the template.",
            file=sys.stderr,
        )
        return 1

    cases = load_eval_cases()
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Running {len(cases)} eval case(s)...")

    metrics = {} if args.skip_llm else build_metrics()
    details: list[dict] = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        result = run_case(case)
        scores = {} if args.skip_llm else score_case(metrics, case, result)
        routing = routing_check(case, result)

        details.append(
            {
                "id": case["id"],
                "input": case["input"],
                "language": case.get("language"),
                "actual_output": result["actual_output"],
                "expected_output": case["expected_output"],
                "predicted_sources": result["sources"],
                "expected_sources": case["expected_sources"],
                "routing": routing,
                "scores": scores,
                "run_id": result["run_id"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        mark = "PASS" if routing["pass"] else "FAIL"
        avg = (
            ""
            if args.skip_llm
            else f" | avg score {sum(v['score'] for v in scores.values()) / len(scores):.3f}"
        )
        print(f"routing={mark}{avg}")

    summary = summarize(details)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(details),
        "judge_model": None if args.skip_llm else settings.gemini_judge_model,
        "answer_model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "skipped_llm_metrics": args.skip_llm,
        **summary,
        "cases": details,
    }

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path = settings.eval_report_path
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Summary ===")
    for name, stats in summary["metrics"].items():
        print(
            f"  {name:<22} avg={stats['average_score']:.3f}  "
            f"pass={stats['pass_count']}/{stats['pass_count'] + stats['fail_count']}"
        )
    ra = summary["routing_accuracy"]
    print(
        f"  {'routing_accuracy':<22} {ra['correct']}/{ra['total']} ({ra['overall']:.0%})"
    )
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
