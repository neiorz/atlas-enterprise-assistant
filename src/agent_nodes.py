"""LangGraph node functions — agentic side (FR-D1, FR-F, FR-G).

    agent <-> tools -> generate_answer / fallback

Split from src/graph_nodes.py (input side) to satisfy NFR-C1's ~250-line
per-file guideline. The agent decides whether a typed tool helps; tools_node
re-validates arguments against the tool's Pydantic model and feeds a failure
back to the agent for exactly one retry (FR-F2).
"""

from __future__ import annotations

from pydantic import ValidationError

from src.llm_setup import llm, llm_with_tools
from src.logging_setup import log_event
from src.prompt_context import format_chunks, format_tool_results, history_as_text
from src.prompts import ANSWER_SYSTEM_PROMPT, TOOL_DECISION_SYSTEM_PROMPT
from src.settings import settings
from src.state import AtlasState
from src.tools import TOOL_REGISTRY


def agent_node(state: AtlasState) -> dict:
    """Decide whether a tool call would help; if so, stage it for tools_node (FR-F3).

    Force-answers when the per-run tool budget is already spent (FR-D5) or when
    the LLM chooses not to call a tool — most questions answer from retrieval alone.
    """
    step = state.get("step_count", 0) + 1

    if state.get("tool_call_count", 0) >= settings.max_tool_calls:
        return {"step_count": step, "force_answer": True}

    retry_note = ""
    prior = state.get("tool_calls", [])
    if prior and prior[-1].get("error"):
        retry_note = (
            f"\n\nYour previous tool call failed validation: {prior[-1]['error']}\n"
            f"Fix the arguments, or don't call a tool if you can't."
        )

    prompt = (
        f"{TOOL_DECISION_SYSTEM_PROMPT}{retry_note}\n\n"
        f"Recent conversation:\n{history_as_text(state.get('chat_history', []))}\n\n"
        f"Retrieved context:\n{format_chunks(state.get('retrieved_chunks', []))}\n\n"
        f"Question: {state['question']}"
    )

    response = llm_with_tools.invoke(prompt)
    calls = getattr(response, "tool_calls", None) or []

    if not calls:
        return {"step_count": step, "force_answer": True}

    call = calls[0]  # one tool per visit; the agent loops if more are needed
    return {
        "step_count": step,
        "force_answer": False,
        "pending_tool_name": call["name"],
        "pending_tool_args": call["args"],
    }


def tools_node(state: AtlasState) -> dict:
    """Execute the pending tool call, re-validating args against its Pydantic
    input model. A validation failure gets exactly one retry (FR-F2), then the
    agent must answer with whatever it already has — the app never crashes."""
    step = state.get("step_count", 0) + 1
    tool_name = state["pending_tool_name"]
    tool_args = state.get("pending_tool_args", {})
    input_model, fn = TOOL_REGISTRY[tool_name]
    tool_calls = list(state.get("tool_calls", []))

    try:
        validated = input_model(**tool_args)
        result = fn(validated)
        tool_calls.append(
            {
                "tool": tool_name,
                "args": tool_args,
                "result": result.model_dump(),
                "error": None,
            }
        )
        log_event(
            state["run_id"],
            state["session_id"],
            "tool_call",
            tool=tool_name,
            args=tool_args,
            success=True,
        )
        return {
            "step_count": step,
            "tool_calls": tool_calls,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "pending_tool_name": None,
        }
    except ValidationError as e:
        error_text = str(e)
        tool_calls.append(
            {"tool": tool_name, "args": tool_args, "result": None, "error": error_text}
        )
        log_event(
            state["run_id"],
            state["session_id"],
            "tool_call",
            tool=tool_name,
            args=tool_args,
            success=False,
            error=error_text,
        )
        retry_count = state.get("validation_retry_count", 0)
        if retry_count >= 1:
            # Second failure: stop retrying and answer with what we have (FR-D5).
            return {
                "step_count": step,
                "tool_calls": tool_calls,
                "pending_tool_name": None,
                "force_answer": True,
            }
        # First failure: clear the pending call so routing loops back to the
        # agent, which sees the error and gets one more attempt.
        return {
            "step_count": step,
            "tool_calls": tool_calls,
            "validation_retry_count": retry_count + 1,
            "pending_tool_name": None,
        }


def answer_node(state: AtlasState) -> dict:
    """Compose the final cited answer in the user's language (FR-G1..G4)."""
    step = state.get("step_count", 0) + 1
    prompt = (
        f"{ANSWER_SYSTEM_PROMPT}\n\n"
        f"Recent conversation:\n{history_as_text(state.get('chat_history', []))}\n\n"
        f"Retrieved context:\n{format_chunks(state.get('retrieved_chunks', []))}"
        f"{format_tool_results(state.get('tool_calls', []))}\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)
    answer_text = response.content

    log_event(
        state["run_id"],
        state["session_id"],
        "final_answer",
        answer=answer_text,
        sources=state.get("sources", []),
    )
    return {"answer": answer_text, "step_count": step}
