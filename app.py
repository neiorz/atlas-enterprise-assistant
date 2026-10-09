"""
Chainlit chat UI (FR-H).

Launch: `chainlit run app.py` (single command, FR-H1).
"""

from __future__ import annotations

import asyncio
import uuid

import chainlit as cl

from src.graph import run_turn


def _welcome_text() -> str:
    return (
        "**Atlas Industries — Internal Knowledge Assistant**\n\n"
        "Ask me about HR, IT, or Finance policies — in English or Arabic. "
        "Use the **New Chat** button below to start a fresh conversation."
    )


def _api_key_ready() -> bool:
    """True only when a real key is configured (not the .env.example placeholder)."""
    from src.settings import settings

    key = settings.google_api_key.strip()
    return bool(key) and not key.startswith("your-")


@cl.on_chat_start
async def on_chat_start() -> None:
    """Assign a fresh session ID (== LangGraph thread_id) for this chat."""
    cl.user_session.set("session_id", str(uuid.uuid4()))
    if not _api_key_ready():
        # NFR-D4 spirit: a clear, actionable message instead of a traceback
        # (or a cryptic Gemini 400) on the first question.
        await cl.Message(
            content=(
                "⚠️ **Missing API key.**\n\n"
                "1. Get a free key at <https://aistudio.google.com/apikey>\n"
                "2. Open `.env` and set `GOOGLE_API_KEY=AIza...`\n"
                "3. Restart the app.\n\n"
                "*(`.env` is gitignored — your key is never committed.)*"
            )
        ).send()
        return
    await cl.Message(content=_welcome_text()).send()


@cl.action_callback("new_chat")
async def on_new_chat(action: cl.Action) -> None:
    """FR-H4: a clearly labelled control that starts a fresh session with
    empty memory (FR-E3) instead of requiring an app restart."""
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await action.remove()
    await cl.Message(
        content="🆕 **New chat started — memory cleared.**\n\n" + _welcome_text()
    ).send()
    await cl.Message(
        content="",
        actions=[
            cl.Action(
                name="new_chat",
                label="🆕 New Chat",
                payload={},
                tooltip="Start a fresh session",
            )
        ],
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Run one turn through the graph and render the workflow + answer."""
    session_id = cl.user_session.get("session_id")

    # NFR-B3: run the (blocking) graph in a worker thread so Chainlit keeps
    # streaming step updates instead of freezing the UI while it works.
    async with cl.Step(name="Routing…", type="run") as step:
        try:
            result = await asyncio.to_thread(run_turn, message.content, session_id)
        # BLE001 is silenced deliberately: this is the UI boundary, and NFR-D3
        # requires that ANY failure (LLM rate limit, network, index missing)
        # surfaces as a friendly message instead of a stack trace. Narrowing
        # the except would let an unhandled error type reach the user.
        except Exception as exc:  # noqa: BLE001 — NFR-D3: never show a stack trace
            await cl.Message(
                content=(
                    "⚠️ Something went wrong running that question "
                    f"(`{type(exc).__name__}`). Please try again in a moment."
                )
            ).send()
            return
        step.output = f"Domain: **{result.get('domain', '?')}**"

    sources = result.get("sources", [])
    if sources:
        async with cl.Step(name="Retrieved sources", type="tool") as step:
            step.output = "\n".join(f"- {s}" for s in sources)

    tool_calls = result.get("tool_calls", [])
    for tc in tool_calls:
        async with cl.Step(name=f"Tool: {tc['tool']}", type="tool") as step:
            if tc.get("error"):
                step.output = f"Args: {tc['args']}\n\nValidation error: {tc['error']}"
            else:
                step.output = f"Args: {tc['args']}\n\nResult: {tc['result']}"

    # FR-H3: Chainlit renders Arabic text RTL automatically based on Unicode
    # bidi rules — no extra markup needed, but verify with an Arabic eval
    # question before submission per the brief.
    await cl.Message(
        content=result.get("answer", ""),
        actions=[
            cl.Action(
                name="new_chat",
                label="🆕 New Chat",
                payload={},
                tooltip="Start a fresh session",
            )
        ],
    ).send()
