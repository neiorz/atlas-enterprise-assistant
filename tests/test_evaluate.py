"""Offline tests for tests/evaluate.py — no API key, no LLM calls.

evaluate.py is the largest deliverable in the repo and would otherwise only
ever run once someone has already spent quota on it. These tests exercise its
real code paths with the graph and judge stubbed, and assert the report shape
FR-J6 requires: per-metric averages, pass/fail counts, routing accuracy, and
per-case detail.

    pytest tests/test_evaluate.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_evaluate_module():
    """Import tests/evaluate.py as a module (it is not in a package)."""
    spec = importlib.util.spec_from_file_location(
        "atlas_evaluate", ROOT / "tests" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate():
    return _load_evaluate_module()


@pytest.fixture
def stubbed(evaluate, monkeypatch, tmp_path):
    """Stub the LLM-facing halves of evaluate.py, redirect the report file."""
    report_path = tmp_path / "eval_report.json"
    monkeypatch.setattr(evaluate.settings, "eval_report_path", report_path)

    # Pass the API-key guard without needing a real key.
    monkeypatch.setattr(evaluate.settings, "google_api_key", "AIzaSTUBFORTESTSONLY")

    # Deterministic graph output, derived per case (no iterator to exhaust).
    def fake_run_case(case):
        arabic = case.get("language") == "ar"
        return {
            "actual_output": (
                f"إجابة عربية للسؤال {case['id']}"
                if arabic
                else f"Answer for {case['id']} citing {case['expected_sources'][0]}"
            ),
            "sources": case["expected_sources"][:1],
            "retrieval_context": ["[Source: x] retrieved chunk text"],
            "router_domain": case["expected_domain"],
            "router_confidence": 0.9,
            "fallback_applied": False,
            "run_id": f"run-{case['id']}",
            "elapsed_seconds": 1.0,
        }

    class StubMetric:
        """Mimics a DeepEval metric after measure(): score/success/reason."""

        def __init__(self, threshold=0.5, **_kwargs):
            self.threshold = threshold
            self.calls = 0

        def measure(self, test_case, **_kwargs):
            self.calls += 1
            # deterministic, threshold-straddling scores
            self.score = 0.4 if self.calls % 2 == 0 else 0.9
            self.success = self.score >= self.threshold
            self.reason = "stub verdict"
            return self.score

    def fake_build_metrics():
        return {name: StubMetric() for name in evaluate.METRIC_NAMES}

    monkeypatch.setattr(evaluate, "run_case", fake_run_case)
    monkeypatch.setattr(evaluate, "build_metrics", fake_build_metrics)
    return report_path


def test_loads_all_ten_gold_cases(evaluate):
    """FR-J1: reads tests/eval_cases.jsonl — the brief requires exactly 10."""
    cases = evaluate.load_eval_cases()
    assert len(cases) == 10
    required = {"id", "input", "expected_output", "expected_domain", "expected_sources"}
    for case in cases:
        assert required <= set(case), f"{case.get('id')} missing {required - set(case)}"
    assert sum(1 for c in cases if c.get("language") == "ar") == 3


def test_gold_context_loads_for_every_case(evaluate):
    """FR-J2: context = full text of expected sources; must not raise for any
    of the 10 cases (a missing file means the corpus wasn't extracted)."""
    for case in evaluate.load_eval_cases():
        texts = evaluate.load_gold_context(case["expected_sources"])
        assert len(texts) == len(case["expected_sources"])
        assert all(len(t) > 200 for t in texts), case["id"]


def test_gold_context_flags_a_missing_source(evaluate):
    with pytest.raises(FileNotFoundError, match="not found"):
        evaluate.load_gold_context(["NOPE-999-does-not-exist.md"])


def test_api_key_guard_rejects_the_placeholder(evaluate, monkeypatch):
    monkeypatch.setattr(evaluate.settings, "google_api_key", "")
    assert evaluate._api_key_ready() is False
    monkeypatch.setattr(evaluate.settings, "google_api_key", "your-google-api-key-here")
    assert evaluate._api_key_ready() is False, "untouched .env.example must not pass"
    monkeypatch.setattr(evaluate.settings, "google_api_key", "AIzaRealLookingKey123")
    assert evaluate._api_key_ready() is True


def test_routing_check_pass_and_fail(evaluate):
    case = {"expected_domain": "finance"}
    ok = evaluate.routing_check(
        case, {"router_domain": "finance", "router_confidence": 0.9}
    )
    assert ok["pass"] is True and ok["predicted_domain"] == "finance"

    # A low-confidence fallback to "multi" is not the right domain (FR-J4)
    fb = evaluate.routing_check(
        {"expected_domain": "hr"},
        {"router_domain": "multi", "router_confidence": 0.4, "fallback_applied": True},
    )
    assert fb["pass"] is False and fb["fallback_applied"] is True


def test_full_run_writes_a_valid_report(evaluate, stubbed, monkeypatch, capsys):
    """FR-J6 end-to-end: main() must produce outputs/eval_report.json with the
    required aggregates plus per-case detail."""
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    assert evaluate.main() == 0

    report = json.loads(stubbed.read_text(encoding="utf-8"))

    # --- FR-J6: per-metric averages + pass/fail counts ---
    assert set(report["metrics"]) == set(evaluate.METRIC_NAMES)
    for stats in report["metrics"].values():
        assert set(stats) == {"average_score", "pass_count", "fail_count", "threshold"}
        assert 0.0 <= stats["average_score"] <= 1.0
        assert stats["pass_count"] + stats["fail_count"] == report["num_cases"]

    # --- routing accuracy (FR-J4) ---
    routing = report["routing_accuracy"]
    assert routing["total"] == report["num_cases"]
    assert routing["correct"] == routing["total"]
    assert routing["overall"] == 1.0

    # --- per-case detail ---
    assert report["num_cases"] == 10
    assert len(report["cases"]) == 10
    for case in report["cases"]:
        assert {
            "id",
            "input",
            "actual_output",
            "expected_output",
            "routing",
            "scores",
        } <= set(case)
        assert set(case["scores"]) == set(evaluate.METRIC_NAMES)
        for stats in case["scores"].values():
            assert {"score", "pass", "reason"} <= set(stats)

    # --- provenance, for the final report's tables ---
    assert report["answer_model"]
    assert report["embedding_model"]

    out = capsys.readouterr().out
    assert "=== Summary ===" in out
    assert "routing_accuracy" in out


def test_skip_llm_writes_routing_only_report(evaluate, stubbed, monkeypatch, capsys):
    """--skip-llm is the offline mode: still a valid report, no judge calls."""
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--limit", "3", "--skip-llm"])
    assert evaluate.main() == 0

    report = json.loads(stubbed.read_text(encoding="utf-8"))
    assert report["num_cases"] == 3
    assert report["metrics"] == {}
    assert report["skipped_llm_metrics"] is True
    assert report["routing_accuracy"]["total"] == 3
    assert all(case["scores"] == {} for case in report["cases"])
    assert "judge_model" in report


def test_limit_flag_truncates_cases(evaluate, stubbed, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--limit", "2", "--skip-llm"])
    evaluate.main()
    report = json.loads(stubbed.read_text(encoding="utf-8"))
    assert report["num_cases"] == 2


def test_main_fails_cleanly_without_a_key(evaluate, monkeypatch, capsys):
    """No traceback — an actionable message and exit 1 (NFR-D4 spirit)."""
    monkeypatch.setattr(evaluate.settings, "google_api_key", "your-google-api-key-here")
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    assert evaluate.main() == 1
    err = capsys.readouterr().err
    assert "aistudio.google.com/apikey" in err
    assert "Traceback" not in err


def test_main_fails_cleanly_when_the_corpus_is_missing(evaluate, monkeypatch, tmp_path):
    """A missing eval_cases.jsonl must say how to fix it (FR-J1/FR-D4)."""
    monkeypatch.setattr(evaluate.settings, "google_api_key", "AIzaSTUBFORTESTSONLY")
    monkeypatch.setattr(evaluate.settings, "eval_cases_path", tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError, match="atlas-corpus.zip"):
        evaluate.load_eval_cases()


def test_summarize_handles_an_empty_run(evaluate):
    """Degenerate input must not divide by zero."""
    summary = evaluate.summarize([])
    assert summary["metrics"] == {}
    assert summary["routing_accuracy"] == {"overall": 0.0, "correct": 0, "total": 0}


def test_stubbed_metrics_are_exercised_once_per_case(evaluate, stubbed, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--limit", "2"])
    evaluate.main()
    report = json.loads(stubbed.read_text(encoding="utf-8"))
    # 5 metrics x 2 cases
    assert len(report["cases"]) == 2
    assert all(
        len(case["scores"]) == len(evaluate.METRIC_NAMES) for case in report["cases"]
    )


def test_module_exports_the_required_metric_names(evaluate):
    """FR-J3 names the exact five DeepEval metrics."""
    assert set(evaluate.METRIC_NAMES) == {
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "hallucination",
    }
    assert set(evaluate.METRIC_CLASSES) == set(evaluate.METRIC_NAMES)


def test_report_is_json_with_ensure_ascii_off(evaluate, stubbed, monkeypatch):
    """Arabic must survive into the report verbatim (readable by graders)."""
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--skip-llm"])
    evaluate.main()
    raw = stubbed.read_text(encoding="utf-8")
    assert "سياسة" in raw or "الإجازات" in raw  # from an Arabic gold case
    assert "\\u0633" not in raw  # not escaped


def test_run_case_reads_router_choice_from_the_run_log(evaluate, monkeypatch, tmp_path):
    """The router's raw choice (not the post-fallback domain) drives FR-J4."""
    log = tmp_path / "run_logs.jsonl"
    monkeypatch.setattr(evaluate.settings, "run_logs_path", log)
    log.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "event": "router_decision",
                "chosen_domain": "hr",
                "confidence": 0.6,
                "fallback_applied": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event = evaluate.find_router_decision("run-1")
    assert event["chosen_domain"] == "hr"
    assert evaluate.find_router_decision("no-such-run") is None


def test_run_case_survives_a_corrupt_log_line(evaluate, monkeypatch, tmp_path):
    """One bad line must not abort a 10-case evaluation."""
    log = tmp_path / "run_logs.jsonl"
    monkeypatch.setattr(evaluate.settings, "run_logs_path", log)
    log.write_text(
        "not json at all\n"
        + json.dumps({"run_id": "r", "event": "router_decision", "chosen_domain": "it"})
        + "\n",
        encoding="utf-8",
    )
    assert evaluate.find_router_decision("r")["chosen_domain"] == "it"


def test_missing_log_returns_none_rather_than_crashing(evaluate, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate.settings, "run_logs_path", tmp_path / "absent.jsonl")
    assert evaluate.find_router_decision("anything") is None
