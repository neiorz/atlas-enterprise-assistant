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
    return settings.has_any_llm_key()


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
        # One metric blowing up must not throw away the whole run: a full
        # evaluation costs many minutes of free-tier quota, and re-running it
        # because the judge returned malformed JSON once is a poor trade. The
        # failure is recorded and summarised separately instead of averaged in.
        try:
            metric.measure(test_case)
        except Exception as exc:  # noqa: BLE001 - record, don't abort (NFR-D3)
            scores[name] = {
                "score": None,
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "reason": "",
            }
            continue
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
    errored = 0
    for name in METRIC_NAMES:
        values = [
            c["scores"][name] for c in cases_detail if name in c.get("scores", {})
        ]
        if not values:
            continue
        # Metrics that failed to run are reported, never averaged in — a
        # score of 0 and "the judge returned invalid JSON" are different
        # claims, and the report must not conflate them.
        errored += sum(1 for v in values if v.get("error"))
        scored = [v for v in values if v.get("score") is not None]
        if not scored:
            metrics_summary[name] = {
                "average_score": None,
                "pass_count": 0,
                "fail_count": 0,
                "errors": len(values),
                "threshold": 0.5,
            }
            continue
        passed = sum(1 for v in scored if v["pass"])
        metrics_summary[name] = {
            "average_score": round(sum(v["score"] for v in scored) / len(scored), 4),
            "pass_count": passed,
            "fail_count": len(scored) - passed,
            "threshold": 0.5,
        }
        if len(scored) != len(values):
            metrics_summary[name]["errors"] = len(values) - len(scored)

    routing = [c["routing"] for c in cases_detail if "routing" in c]
    routing_pass = sum(1 for r in routing if r["pass"])
    failed = [c for c in cases_detail if c.get("error")]
    return {
        "metrics": metrics_summary,
        "metric_errors": errored,
        "errored_cases": len(failed),
        "errored_case_ids": [c["id"] for c in failed],
        "routing_accuracy": {
            "overall": round(routing_pass / len(routing), 4) if routing else 0.0,
            "correct": routing_pass,
            "total": len(routing),
        },
    }


# Groq's free tier refuses a request when the rolling per-day budget is full
# and tells you exactly when to come back ("try again in 1m20s") — that is when
# enough old usage has expired to fit the request. Waiting it out is the
# difference between a 10-case evaluation and a 0-case one: a naive run once
# lost all ten results to this. Only throttling is retried; a genuinely broken
# case should fail fast instead of burning three minutes of quota.
CASE_RETRIES = 3
RETRY_BACKOFF_S = (75, 150, 300)


def _is_throttling(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "Rate limit" in text or "rate_limit_exceeded" in text


def _case_with_retries(
    case: dict, *, metrics: dict, skip_llm: bool
) -> tuple[dict, dict, dict]:
    """Run one gold case, waiting out free-tier throttling between attempts.

    Returns ``(result, scores, routing)`` — the same triple the caller builds
    on success. Re-raises the last exception when every attempt fails so the
    caller can record it and move on to the next case.
    """
    last: Exception | None = None
    for attempt in range(1, CASE_RETRIES + 1):
        try:
            result = run_case(case)
            scores = {} if skip_llm else score_case(metrics, case, result)
            return result, scores, routing_check(case, result)
        except Exception as exc:
            last = exc
            if attempt == CASE_RETRIES or not _is_throttling(exc):
                raise
            wait = RETRY_BACKOFF_S[attempt - 1]
            print(
                f"throttled — waiting {wait}s "
                f"(attempt {attempt}/{CASE_RETRIES})... ",
                end="",
                flush=True,
            )
            time.sleep(wait)
    raise last if last else RuntimeError("unreachable")


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
            "ERROR: no LLM API key is set (or it is still the placeholder).\n"
            "  Set EITHER of these in .env — both have free tiers:\n"
            "    GROQ_API_KEY=...      https://console.groq.com/keys\n"
            "    GOOGLE_API_KEY=AIza…  https://aistudio.google.com/apikey\n"
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
        # Free-tier LLMs 429 on both a per-minute and a rolling per-day budget,
        # and a single failed case must not discard the other nine — an
        # evaluation costs many minutes of quota and this is the only place
        # the report is written. So: retry the *case* with enough backoff for
        # the rolling window to free tokens, then record the failure and keep
        # going. summarize() reports errored cases separately instead of
        # averaging them in, so a partial run is never mistaken for a full one.
        try:
            result, scores, routing = _case_with_retries(
                case, metrics=metrics, skip_llm=args.skip_llm
            )
        except Exception as exc:  # noqa: BLE001 — record, don't abort
            details.append(
                {
                    "id": case["id"],
                    "input": case["input"],
                    "language": case.get("language"),
                    "expected_output": case["expected_output"],
                    "expected_sources": case["expected_sources"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"ERROR ({type(exc).__name__}) — skipped, continuing")
            continue

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
        scored = [v["score"] for v in scores.values() if v.get("score") is not None]
        if args.skip_llm or not scores:
            avg = ""
        elif not scored:
            avg = " | all metrics errored"
        else:
            avg = f" | avg score {sum(scored) / len(scored):.3f}"
            if len(scored) != len(scores):
                avg += f" ({len(scores) - len(scored)} errored)"
        print(f"routing={mark}{avg}")

    summary = summarize(details)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(details),
        "provider": settings.resolved_provider,
        "judge_model": None if args.skip_llm else settings.active_judge_model,
        "answer_model": settings.active_model,
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
        avg = (
            "n/a" if stats["average_score"] is None else f"{stats['average_score']:.3f}"
        )
        print(
            f"  {name:<22} avg={avg}  "
            f"pass={stats['pass_count']}/{stats['pass_count'] + stats['fail_count']}"
        )
    ra = summary["routing_accuracy"]
    print(
        f"  {'routing_accuracy':<22} {ra['correct']}/{ra['total']} ({ra['overall']:.0%})"
    )
    if summary["errored_cases"]:
        # Surfaced so a partially-completed run is never mistaken for a
        # complete one when reading outputs/eval_report.json.
        ids = ", ".join(summary["errored_case_ids"])
        print(f"  {'errored cases':<22} {summary['errored_cases']} ({ids})")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
