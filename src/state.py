"""
Graph state schema (FR-D3).

Fields with no reducer follow LangGraph's default "last write wins" — exactly
what we want for per-turn fields (domain, retrieved_chunks, answer, ...),
which src/graph.py's memory_loader node explicitly resets at the start of
every turn anyway. chat_history is the one field that must survive across
turns within a session: it's Annotated with operator.add so that, combined
with the MemorySaver checkpointer keyed by session_id (thread_id), each
turn's Q/A pair is appended to — never overwrites — the running conversation.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Domain = Literal["hr", "it", "finance"]


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AtlasState(TypedDict, total=False):
    # --- Per-turn input ---
    question: str
    session_id: str
    run_id: str

    # --- Routing (FR-D2) ---
    domain: str
    router_confidence: float

    # --- Retrieval ---
    retrieved_chunks: list[dict]
    sources: list[str]

    # --- Tooling (FR-F) ---
    tool_calls: list[dict]
    tool_call_count: int
    validation_retry_count: int
    pending_tool_name: str | None
    pending_tool_args: dict

    # --- Control (FR-D4, FR-D5) ---
    step_count: int
    force_answer: bool

    # --- Output ---
    answer: str

    chat_history: Annotated[list[ChatTurn], operator.add]


class RouterDecision(BaseModel):
    domain: Domain
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
