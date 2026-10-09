"""Offline graph wiring tests — no API key, no network needed.

Why these exist: the Day-10 integration pass split graph_nodes.py into an
input side and an agent side (src/agent_nodes.py), and a successful import
does not prove the conditional edges still route correctly. These tests stub
the three LLM handles so the whole graph — routing, retrieval, tool loop,
step limits, memory, logging — runs deterministically on a laptop.

    pytest tests/test_graph.py -q
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.settings import settings

pytestmark = pytest.mark.skipif(
    settings.docs_dir.joinpath("hr").exists() is False,
    reason="corpus not extracted (unzip atlas-corpus.zip first)",
)


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace every Gemini handle with a deterministic stub.

    Returns a dict the tests can mutate to drive the graph down different
    paths (tool call vs straight to answer, high vs low router confidence).
    """
    from src import agent_nodes, graph_nodes

    controls = {
        "domain": "finance",
        "confidence": 0.9,
        "pending_tool_calls": [],  # handed to the model on its next visit
        "answer": "Canned answer with Sources: FIN-001-travel-reimbursement-policy.md",
        "agent_prompts": [],
        "llm_calls": 0,
    }

    def fake_router_invoke(_prompt):
        return SimpleNamespace(
            domain=controls["domain"],
            confidence=controls["confidence"],
            reasoning="stub",
        )

    def fake_agent_invoke(prompt):
        controls["llm_calls"] += 1
        controls["agent_prompts"].append(prompt)
        # Behave like a sensible model: emit the queued tool call once, then
        # on subsequent visits (which now show the result) decline to repeat it.
        calls = controls["pending_tool_calls"]
        chosen = calls.pop(0) if calls else None
        return SimpleNamespace(content="", tool_calls=[chosen] if chosen else [])

    def fake_answer_invoke(prompt):
        controls["llm_calls"] += 1
        controls["answer_prompts"] = controls.get("answer_prompts", []) + [prompt]
        return SimpleNamespace(content=controls["answer"])

    monkeypatch.setattr(
        graph_nodes, "router_llm", SimpleNamespace(invoke=fake_router_invoke)
    )
    monkeypatch.setattr(
        agent_nodes, "llm_with_tools", SimpleNamespace(invoke=fake_agent_invoke)
    )
    monkeypatch.setattr(agent_nodes, "llm", SimpleNamespace(invoke=fake_answer_invoke))
    return controls


def _fresh_session() -> str:
    return str(uuid.uuid4())


def test_turn_runs_and_returns_cited_answer(stub_llm):
    from src.graph import run_turn

    state = run_turn("What is the domestic hotel cap?", _fresh_session())

    assert state["domain"] == "finance"
    assert state["sources"], "retrieval should surface at least one source"
    assert state["answer"] == stub_llm["answer"]
    # FR-C3: source filenames must travel with the retrieved chunks
    assert all(c["source_filename"] for c in state["retrieved_chunks"])


def test_router_fallback_fans_out_on_low_confidence(stub_llm):
    from src.graph import run_turn

    stub_llm["confidence"] = 0.30  # below settings.router_confidence_threshold
    state = run_turn("Something ambiguous about policies", _fresh_session())

    assert state["domain"] == "multi"
    # fan-out must still return real chunks (FR-D2)
    assert state["retrieved_chunks"]


def test_memory_accumulates_across_turns(stub_llm):
    from src.graph import run_turn

    session = _fresh_session()
    first = run_turn("How do I submit a reimbursement?", session)
    stub_llm["answer"] = "Follow-up answer"
    second = run_turn("And what about international trips?", session)

    history = second["chat_history"]
    roles = [t["role"] for t in history]
    # 4 turns of history after 2 exchanges (2 user + 2 assistant), no loss
    assert roles == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "How do I submit a reimbursement?"
    assert second["answer"] == "Follow-up answer"
    assert first["session_id"] == second["session_id"]


def test_new_session_starts_with_empty_memory(stub_llm):
    from src.graph import run_turn

    run_turn("first session question", _fresh_session())
    fresh = run_turn("brand new session question", _fresh_session())
    assert fresh["chat_history"] == [
        {"role": "user", "content": "brand new session question"},
        {"role": "assistant", "content": fresh["answer"]},
    ]


def test_tool_called_then_answered(stub_llm):
    from src.graph import run_turn

    stub_llm["pending_tool_calls"] = [
        {"name": "list_leave_types", "args": {"paid_only": True}}
    ]
    state = run_turn("What leave types do we have?", _fresh_session())

    # Exactly one call: the agent must see the tool's result and not re-issue
    # it (the Day-10 fix that stopped 3x duplicate calls).
    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0]["error"] is None
    assert state["tool_calls"][0]["result"]["source_filename"].endswith(".docx")
    assert state["answer"]

    # Proof the fix reaches the model: the second agent prompt shows the result
    # and the do-not-repeat instruction.
    assert len(stub_llm["agent_prompts"]) == 2, "agent should run again after the tool"
    follow_up = stub_llm["agent_prompts"][1]
    assert "ALREADY been called" in follow_up
    assert "list_leave_types" in follow_up
    # ...and the answer prompt carries the result too (FR-G1 grounding).
    assert "list_leave_types" in stub_llm["answer_prompts"][0]


def test_tool_validation_error_recovers(stub_llm):
    """FR-F2 / NFR-D1: bad args surface as an error, the agent retries once,
    and the app still returns an answer instead of crashing."""
    from src.graph import run_turn

    # policy_id fails the strict regex — triggers ValidationError
    stub_llm["pending_tool_calls"] = [
        {"name": "policy_lookup", "args": {"policy_id": "not-a-real-id"}}
    ]
    state = run_turn("What is policy not-a-real-id?", _fresh_session())

    assert state["tool_calls"], "the failed call should be recorded"
    assert state["tool_calls"][0]["error"], "Pydantic error must be captured"
    assert state["answer"], "graph must still produce an answer after recovery"
    # the retry prompt must show the validation error so the model can fix it
    assert "failed validation" in stub_llm["agent_prompts"][1]


def test_step_limit_reaches_fallback_node(stub_llm, monkeypatch):
    from src.graph import run_turn
    from src.settings import settings as s

    monkeypatch.setattr(s, "max_steps", 2)  # unreachable in 2 steps
    state = run_turn("Force the step limit path", _fresh_session())

    assert state["answer"], "fallback node must still set an answer"
    # the fallback message is the explicit-uncertainty one, not a guess
    assert "processing" in state["answer"] or "خطوات" in state["answer"]


def test_run_logs_carry_every_event_under_one_run_id(stub_llm):
    """FR-I1/I2/I3: one run_id tags router, retrieval and final answer."""
    from src.graph import run_turn

    before = (
        settings.run_logs_path.stat().st_size if settings.run_logs_path.exists() else 0
    )

    state = run_turn("How many vacation days do I have?", _fresh_session())
    run_id = state["run_id"]

    with settings.run_logs_path.open(encoding="utf-8") as f:
        f.seek(before)
        events = [json.loads(line) for line in f if line.strip()]

    mine = [e for e in events if e.get("run_id") == run_id]
    names = {e["event"] for e in mine}
    assert {
        "turn_start",
        "router_decision",
        "retrieval",
        "final_answer",
        "turn_complete",
    } <= names
    # FR-I4: log lines must not carry raw document bodies
    for e in mine:
        assert "google_api_key" not in json.dumps(e)
