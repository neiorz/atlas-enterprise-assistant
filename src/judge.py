"""Judge LLM for DeepEval (FR-J5).

DeepEval's metrics accept any object implementing ``DeepEvalBaseLLM`` (the
class is spelled with uppercase "LLM" from DeepEval 2.x onward — older
tutorials showing ``DeepEvalBaseLlm`` no longer import). We wrap our own
answer-path client rather than relying on DeepEval's default OpenAI judge
because:

1. FR-J5 — the judge MUST handle Arabic; 3 of the 10 gold cases are Arabic
   and every metric is scored on them. Both configured backends (Gemini 2.0
   Flash, Qwen3.8-27B) read Arabic natively.
2. NFR-A1 — no second provider or paid key: the judge reuses whichever
   free-tier key already drives the answer path, decided by
   settings.resolved_provider.

The judge gets its own rate-limiter instance (separate from the answer LLM's
in src/llm_setup.py) so the 50 back-to-back metric calls of a full evaluation
never starve a live chat turn, and vice versa.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM

from src.llm_setup import JUDGE_MAX_TOKENS, build_chat_llm, make_rate_limiter
from src.settings import settings

# DeepEval fires metric calls back-to-back (5 metrics x 10 cases = 50 calls),
# so the judge gets its own limiter and cannot starve a live chat turn (or
# vice versa). On Groq the pacing matters much more: free-tier Groq meters
# *output* tokens per minute (~1000) and answers every over-budget request
# with a 429, so we trade a slow, steady evaluation for a thrashing one.
_GROQ = settings.resolved_provider == "groq"
_judge_rate_limiter = make_rate_limiter(
    burst=2 if _GROQ else 5,
    requests_per_second=0.12 if _GROQ else None,
)


class JudgeLLM(DeepEvalBaseLLM):
    """Adapts our LangChain chat model to DeepEval's judge interface."""

    def __init__(self) -> None:
        self._llm = build_chat_llm(
            temperature=0.0,  # judge must be deterministic across runs
            rate_limiter=_judge_rate_limiter,
            max_tokens=JUDGE_MAX_TOKENS,
        )

    # --- required abstract methods -------------------------------------

    def load_model(self) -> JudgeLLM:
        """DeepEval calls this before scoring; the client is built in __init__."""
        return self

    def get_model_name(self) -> str:
        """Recorded in eval_report.json so scores are traceable to a judge."""
        return settings.active_judge_model

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return _as_text(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self._llm.ainvoke(prompt)
        return _as_text(response.content)

    # --- optional helpers (batch + schema variants) ---------------------

    def batch_generate(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]

    async def a_batch_generate(self, prompts: list[str]) -> list[str]:
        return [await self.a_generate(p) for p in prompts]

    def generate_with_schema(self, prompt: str, schema=None) -> str:
        return self.generate(prompt)

    async def a_generate_with_schema(self, prompt: str, schema=None) -> str:
        return await self.a_generate(prompt)

    def supports_json_mode(self) -> bool:
        return False


def _as_text(content) -> str:
    """The model may return a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


_judge: JudgeLLM | None = None


def get_judge() -> JudgeLLM:
    """Lazy singleton — importing tests/evaluate.py must not require an API
    key until we actually start scoring."""
    global _judge
    if _judge is None:
        _judge = JudgeLLM()
    return _judge


# Backwards-compatible alias: the class used to be Gemini-only.
GeminiJudge = JudgeLLM
