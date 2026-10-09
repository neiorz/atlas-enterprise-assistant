"""Gemini-backed judge LLM for DeepEval (FR-J5).

DeepEval's metrics accept any object implementing ``DeepEvalBaseLLM`` (the
class is spelled with uppercase "LLM" from DeepEval 2.x onward — older
tutorials showing ``DeepEvalBaseLlm`` no longer import). We wrap our existing
Gemini client rather than relying on DeepEval's default OpenAI judge because:

1. FR-J5 — the judge MUST handle Arabic; 3 of the 10 gold cases are Arabic
   and every metric is scored on them. Gemini 2.0 Flash reads Arabic natively.
2. NFR-A1 — no second provider or paid key: the whole project runs on the one
   free-tier GOOGLE_API_KEY.

The judge gets its own rate-limiter instance (separate from the answer LLM's
in src/llm_setup.py) so the 50 back-to-back metric calls of a full evaluation
never starve a live chat turn, and vice versa.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import settings

# DeepEval fires metric calls back-to-back (5 metrics x 10 cases = 50 calls).
# 0.25 rps == Gemini free tier's ~15 requests/minute; a burst bucket of 5 lets
# one test case's five metrics flow without waiting between each, while the
# sustained rate still averages out under the free-tier ceiling.
_judge_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,
    check_every_n_seconds=0.1,
    max_bucket_size=5,
)


class GeminiJudge(DeepEvalBaseLLM):
    """Adapts langchain's ChatGoogleGenerativeAI to DeepEval's judge interface."""

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_judge_model,
            temperature=0.0,  # judge must be deterministic across runs
            google_api_key=settings.google_api_key,
            rate_limiter=_judge_rate_limiter,
            max_retries=6,
        )

    # --- required abstract methods -------------------------------------

    def load_model(self) -> GeminiJudge:
        """DeepEval calls this before scoring; the client is built in __init__."""
        return self

    def get_model_name(self) -> str:
        """Recorded in eval_report.json so scores are traceable to a judge."""
        return settings.gemini_judge_model

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
    """Gemini may return a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


_judge: GeminiJudge | None = None


def get_judge() -> GeminiJudge:
    """Lazy singleton — importing tests/evaluate.py must not require an API
    key until we actually start scoring."""
    global _judge
    if _judge is None:
        _judge = GeminiJudge()
    return _judge
