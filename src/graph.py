"""
LangGraph assembly (FR-D).

Node layout (FR-D1):

    memory_loader -> router -> retriever -> agent <-> tools -> answer -> memory_writer

"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent_nodes import agent_node, answer_node, tools_node
from src.graph_nodes import (
    fallback_node,
    memory_loader_node,
    memory_writer_node,
    retriever_node,
    router_node,
)
from src.logging_setup import log_event, new_run_id
from src.settings import settings
from src.state import AtlasState


def _route_after_agent(state: AtlasState) -> str:
    """After agent: fallback on step overflow, else tools or answer."""
    if state.get("step_count", 0) >= settings.max_steps:
        return "fallback"
    if state.get("force_answer"):
        return "generate_answer"
    if state.get("pending_tool_name"):
        return "tools"
    return "generate_answer"


def _route_after_tools(state: AtlasState) -> str:
    """After tools: fallback on step overflow, answer if forced, else loop to agent."""
    if state.get("step_count", 0) >= settings.max_steps:
        return "fallback"
    if state.get("force_answer"):
        return "generate_answer"
    return "agent"


def build_graph():
    """Assemble and compile the Atlas graph with an in-memory checkpointer.

    Returns:
        A compiled LangGraph app. The checkpointer keys state by thread_id
        (== session_id), which is what makes chat_history persist across
        turns of the same chat session (FR-E) without any external DB.
    """
    graph = StateGraph(AtlasState)

    graph.add_node("memory_loader", memory_loader_node)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("memory_writer", memory_writer_node)

    graph.add_edge(START, "memory_loader")
    graph.add_edge("memory_loader", "router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "agent")

    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "tools": "tools",
            "generate_answer": "generate_answer",
            "fallback": "fallback",
        },
    )
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "agent": "agent",
            "generate_answer": "generate_answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate_answer", "memory_writer")
    graph.add_edge("fallback", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=MemorySaver())


atlas_graph = build_graph()


def run_turn(question: str, session_id: str) -> dict:
    """Run one question through the graph (used by app.py and tests/evaluate.py).

    Args:
        question: The employee's question, in English or Arabic.
        session_id: Stable per-chat-session ID; doubles as the LangGraph
            thread_id so memory persists across turns of the same session.

    Returns:
        The final graph state, including "answer" and "sources".
    """
    run_id = new_run_id()
    config = {"configurable": {"thread_id": session_id}}
    log_event(run_id, session_id, "turn_start", question=question)

    return atlas_graph.invoke(
        {"question": question, "session_id": session_id, "run_id": run_id},
        config=config,
    )
