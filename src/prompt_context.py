"""Helpers that render graph state into prompt-ready text blocks.

Split out of graph_nodes.py so both the input-side nodes (router, retriever)
and the agentic nodes (agent, answer) can share them without dragging each
other's code along — this keeps every node file well under NFR-C1's ~250-line
guideline and makes the node layout easier to navigate.
"""

from __future__ import annotations

import json


def history_as_text(history: list[dict], max_turns: int = 6) -> str:
    """Render the last few chat turns as plain text for prompt context.

    Args:
        history: Session chat_history entries ({role, content}).
        max_turns: How many most-recent turns to include, so a long session
            can't blow past the model's context window.

    Returns:
        A "USER: ...\nASSISTANT: ..." block, or a placeholder when empty.
    """
    if not history:
        return "(no prior turns in this session)"
    recent = history[-max_turns:]
    return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in recent)


def format_chunks(chunks: list[dict], max_chars_per_chunk: int | None = None) -> str:
    """Render retrieved chunks as a labeled context block.

    Each block is prefixed with its source filename so the answer prompt can
    tie every fact back to a real document (FR-G1, FR-G2).

    Args:
        max_chars_per_chunk: truncate each chunk to this many characters.
            The agent node passes a small value because it only has to judge
            whether a tool would add anything; the answer node keeps full
            text so it can quote accurately. Every duplicated token costs
            real latency: free-tier Groq refills its token budget at only
            ~133 tokens/second, so re-sending full context twice per turn is
            the single biggest contributor to NFR-B1 overrun.
    """
    if not chunks:
        return "(no relevant documents retrieved)"

    def _render(chunk: dict) -> str:
        text = chunk["text"]
        if max_chars_per_chunk is not None and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " …[truncated]"
        return f"[Source: {chunk['source_filename']}]\n{text}"

    return "\n\n---\n\n".join(_render(c) for c in chunks)


def format_tool_results(tool_calls: list[dict]) -> str:
    """Render tool outcomes for the agent/answer prompts.

    JSON (not Python's repr) so nested results stay readable and Arabic text
    is not escaped to \\uXXXX — plus the error cases, which the agent needs in
    order to retry a failed call (FR-F2).

    Returns an empty string when no tool has run, so callers can concatenate
    unconditionally.
    """
    if not tool_calls:
        return ""
    parts = []
    for tc in tool_calls:
        if tc.get("error"):
            body = f"FAILED validation: {tc['error']}"
        else:
            body = json.dumps(tc.get("result"), ensure_ascii=False)
        parts.append(f"Tool `{tc['tool']}` -> {body}")
    return "\n\nTool results so far:\n" + "\n".join(parts)
