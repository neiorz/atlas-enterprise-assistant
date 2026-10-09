"""
LLM client and tool-binding setup (split out so src/graph_nodes.py stays
focused on node logic, per NFR-C1's ~250-line-per-file guideline).

The client is built at import time and is provider-agnostic: whichever
free-tier key is configured (Gemini or Groq) decides the backend, via
settings.resolved_provider. That resolution never raises, because the
offline test suite imports this module with no credentials at all — the
"you have no API key" message is produced by the guards in app.py and
tests/evaluate.py instead, where it can be shown to a human.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import settings
from src.state import RouterDecision
from src.tools import TOOL_REGISTRY

try:  # Groq is an optional second provider, not a hard dependency.
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - depends on how the venv was built
    ChatGroq = None  # type: ignore[assignment]

# Free-tier ceilings differ sharply between providers, and NFR-B1's 8-second
# per-turn budget is dominated by how fast we may issue the turn's three
# sequential LLM calls (router -> agent -> answer):
#
#   Gemini free tier  ~15 req/min  -> 0.25 rps, burst 3 (exactly one turn)
#   Groq free tier    far higher   -> 1.0 rps,  burst 6 (two turns)
#
# A single limiter pinned to Gemini's ceiling would make a Groq user wait
# ~8s between turns for no reason at all — which on its own fails NFR-B1.
# max_retries=6 on the clients absorbs the occasional 429 either way.
#
# Setting this to 0.06 (one call per ~17s) as originally drafted made every
# turn take a minute; that failed NFR-B1 too.
_RATE_PROFILES: dict[str, tuple[float, int]] = {
    "gemini": (0.25, 3),
    "groq": (1.0, 6),
}


def make_rate_limiter(
    burst: int | None = None, requests_per_second: float | None = None
) -> InMemoryRateLimiter:
    """A limiter sized for the configured provider's free tier.

    Args:
        burst: overrides the provider's default bucket size. The DeepEval
            judge passes a larger burst so one test case's five metrics flow
            without waiting between each.
        requests_per_second: overrides the provider's default sustained rate.
            The judge lowers this because free-tier Groq meters *output*
            tokens per minute (~1000) — pacing requests is the only way to
            stay under it instead of earning a 429 on every other call.
    """
    default_rps, default_burst = _RATE_PROFILES[settings.resolved_provider]
    return InMemoryRateLimiter(
        requests_per_second=(
            default_rps if requests_per_second is None else requests_per_second
        ),
        check_every_n_seconds=0.1,
        max_bucket_size=default_burst if burst is None else burst,
    )


_rate_limiter = make_rate_limiter()

# Output cap per request. Groq's free tier meters output tokens per minute
# (measured limit: 1000 OTPM) and *rejects up front* any request whose
# expected output exceeds it — so an unset max_tokens (which makes the API
# assume ~1145) fails before generating anything.
#
# 500 leaves ample room for a cited answer whose `Sources:` list is the last
# thing generated — truncating that list would silently break FR-G citations,
# which are graded. The judge needs more: DeepEval asks for structured JSON
# and a 320 cap cut it mid-object ("invalid JSON" from the metric).
DEFAULT_MAX_TOKENS = 500
JUDGE_MAX_TOKENS = 800


def build_chat_llm(
    temperature: float | None = None,
    rate_limiter: InMemoryRateLimiter | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Build the chat model for the configured provider.

    Args:
        temperature: overrides settings.llm_temperature (the judge passes 0.0
            so scoring stays deterministic across runs).
        rate_limiter: separate limiter instance for callers that must not
            share the chat path's budget (the DeepEval judge runs 50 calls
            back-to-back and would otherwise starve a live chat turn).
        max_tokens: output cap. This is not optional on Groq's free tier —
            it enforces output tokens per minute (OTPM), and a request whose
            *expected* output exceeds the cap is rejected outright with a
            429 before generating a single token. Leaving it unset makes the
            API assume a large default (measured: 1145 vs a 1000 OTPM limit),
            so every judge call and most answer calls failed.
    """
    temp = settings.llm_temperature if temperature is None else temperature
    limiter = _rate_limiter if rate_limiter is None else rate_limiter
    cap = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    provider = settings.resolved_provider

    if provider == "groq":
        if ChatGroq is None:
            raise RuntimeError(
                "LLM_PROVIDER=groq but langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            )
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=temp,
            max_tokens=cap,
            rate_limiter=limiter,
            max_retries=6,
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temp,
        google_api_key=settings.google_api_key,
        max_output_tokens=cap,
        rate_limiter=limiter,
        max_retries=6,
    )


llm = build_chat_llm()

router_llm = llm.with_structured_output(RouterDecision)

_TOOL_DESCRIPTIONS = {
    "policy_lookup": (
        "Look up which document defines a specific policy/document ID "
        "(e.g. 'FIN-TRA-008', 'HR-VAC-001'). Use only when the user mentions "
        "or asks about a specific ID directly."
    ),
    "reimbursement_calculator": (
        "Compute exact reimbursable amounts against Atlas Industries' caps "
        "for hotel/taxi/client_dinner/per_diem expenses, given a trip_type "
        "and a list of items. Use only when the user gives concrete amounts "
        "and wants an exact reimbursement figure or a cap check."
    ),
    "list_leave_types": (
        "Return the full structured list of Atlas Industries leave types "
        "(paid and unpaid). Use only when the user asks for a full list, "
        "not a single leave type's details."
    ),
}


def _build_langchain_tools() -> list[StructuredTool]:
    """Exposes our Pydantic tool schemas to the LLM for tool-calling.

    Returns:
        LangChain tool wrappers for bind_tools(), one per entry in
        TOOL_REGISTRY. The bound function is never actually invoked —
        real execution always happens in graph_nodes.tools_node against
        TOOL_REGISTRY directly, so our own Pydantic validation and retry
        logic stays in full control regardless of the LLM function-calling
        layer's internals.
    """
    tools = []
    for name, (input_model, _fn) in TOOL_REGISTRY.items():
        tools.append(
            StructuredTool.from_function(
                func=lambda **kwargs: kwargs,
                name=name,
                description=_TOOL_DESCRIPTIONS[name],
                args_schema=input_model,
            )
        )
    return tools


llm_with_tools = llm.bind_tools(_build_langchain_tools())
