"""
LLM client and tool-binding setup (split out so src/graph_nodes.py stays
focused on node logic, per NFR-C1's ~250-line-per-file guideline).
"""

from __future__ import annotations

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import settings
from src.state import RouterDecision
from src.tools import TOOL_REGISTRY

# Gemini free tier allows ~15 requests/minute (0.25 rps). A burst bucket of 3
# lets one graph turn's three sequential LLM calls (router -> agent -> answer)
# fire back-to-back so a warm turn can finish inside NFR-B1's 8-second budget;
# the sustained rate still averages out to the free-tier ceiling, and
# max_retries=6 absorbs the occasional 429. Setting this to 0.06 (one call per
# ~17s) as originally drafted made every turn take a minute — that failed NFR-B1.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,
    check_every_n_seconds=0.1,
    max_bucket_size=3,
)

llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    temperature=settings.llm_temperature,
    google_api_key=settings.google_api_key,
    rate_limiter=_rate_limiter,
    max_retries=6,
)

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
