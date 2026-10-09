"""LangGraph node functions — input side (FR-D1).

    memory_loader -> router -> retriever -> [agent side: src/agent_nodes.py]

Kept separate from the agentic nodes so each file stays well under NFR-C1's
~250-line guideline. Shared prompt-rendering helpers live in src/prompt_context.py.
"""

from __future__ import annotations

from src.chunking import detect_language
from src.llm_setup import router_llm
from src.logging_setup import log_event
from src.prompt_context import history_as_text
from src.prompts import ROUTER_SYSTEM_PROMPT
from src.retriever import retrieve, retrieve_multi_domain, unique_sources
from src.settings import settings
from src.state import AtlasState, RouterDecision


def memory_loader_node(state: AtlasState) -> dict:
    """Reset this turn's transient fields; chat_history is left untouched
    since it persists across turns via the MemorySaver checkpointer."""
    return {
        "step_count": 0,
        "tool_calls": [],
        "tool_call_count": 0,
        "validation_retry_count": 0,
        "force_answer": False,
        "pending_tool_name": None,
        "pending_tool_args": {},
        "retrieved_chunks": [],
        "sources": [],
        "answer": "",
    }


def router_node(state: AtlasState) -> dict:
    """Classify the question into hr/it/finance, or 'multi' on low confidence (FR-D2).

    The raw LLM choice and confidence are logged (even when the fallback kicks
    in) so tests/evaluate.py can score routing accuracy against the router's
    actual decision rather than the post-fallback domain.
    """
    step = state.get("step_count", 0) + 1
    history_text = history_as_text(state.get("chat_history", []))

    decision: RouterDecision = router_llm.invoke(
        f"{ROUTER_SYSTEM_PROMPT}\n\nRecent conversation:\n{history_text}\n\nQuestion: {state['question']}"
    )

    domain = decision.domain
    fallback_applied = decision.confidence < settings.router_confidence_threshold
    if fallback_applied:
        # FR-D2 fallback: rather than guessing one narrow (possibly wrong) domain,
        # fan retrieval out across all three — broader and safer.
        domain = "multi"

    log_event(
        state["run_id"],
        state["session_id"],
        "router_decision",
        chosen_domain=decision.domain,
        confidence=decision.confidence,
        fallback_applied=fallback_applied,
    )
    return {
        "domain": domain,
        "router_confidence": decision.confidence,
        "step_count": step,
    }


def retriever_node(state: AtlasState) -> dict:
    """Pull domain-filtered chunks (or fan out across all domains) for the question."""
    step = state.get("step_count", 0) + 1
    domain = state["domain"]

    if domain == "multi":
        chunks = retrieve_multi_domain(state["question"], list(settings.domains))
    else:
        chunks = retrieve(state["question"], domain)

    chunk_dicts = [vars(c) for c in chunks]
    sources = unique_sources(chunks)

    log_event(
        state["run_id"],
        state["session_id"],
        "retrieval",
        domain=domain,
        num_chunks=len(chunks),
        sources=sources,
    )
    return {"retrieved_chunks": chunk_dicts, "sources": sources, "step_count": step}


def fallback_node(state: AtlasState) -> dict:
    """Reached only when the per-run step limit is exceeded (FR-D4).

    Answers with explicit uncertainty in the question's language instead of
    looping forever — never a stack trace, never a guess.
    """
    step = state.get("step_count", 0) + 1
    language = detect_language(state["question"])
    message = (
        "لم أتمكن من إكمال هذا الطلب ضمن الحد الأقصى لعدد خطوات المعالجة المسموح بها. "
        "برجاء إعادة صياغة السؤال بشكل أبسط."
        if language == "ar"
        else "I wasn't able to complete this within the allowed number of processing "
        "steps. Please try rephrasing your question more simply."
    )
    log_event(
        state["run_id"],
        state["session_id"],
        "fallback_step_limit",
        step_count=state.get("step_count", 0),
    )
    return {"answer": message, "sources": [], "step_count": step}


def memory_writer_node(state: AtlasState) -> dict:
    """Append this turn to chat_history (accumulated via the operator.add reducer)."""
    log_event(
        state["run_id"],
        state["session_id"],
        "turn_complete",
        step_count=state.get("step_count", 0),
    )
    return {
        "chat_history": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state["answer"]},
        ]
    }
