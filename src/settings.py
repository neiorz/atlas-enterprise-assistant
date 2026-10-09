"""
Centralized configuration for the Atlas Enterprise Assistant.

"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Values copied verbatim out of .env.example are placeholders, not credentials.
_UNSET_KEYS = frozenset({"", "your-google-api-key-here", "your-groq-api-key-here"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    docs_dir: Path = data_dir / "docs"
    tests_dir: Path = project_root / "tests"
    outputs_dir: Path = project_root / "outputs"

    # --- Domains ---
    domains: tuple[str, ...] = ("hr", "it", "finance")

    # --- LLM provider ---
    # The assistant runs on whichever free-tier provider has a real key.
    # "auto" prefers Gemini (the team's documented default) and falls back to
    # Groq; set this explicitly to force one. Nothing here is paid: both
    # providers are used on their free tiers, so the project stays at $0.
    llm_provider: str = "auto"

    # --- LLM (Google Gemini) ---
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_judge_model: str = "gemini-2.0-flash"

    # --- LLM (Groq) ---
    # qwen/qwen3.8-27b: ~0.3s per call on Groq's free tier (well inside
    # NFR-B1's 8s), fluent Arabic for the 3 Arabic gold cases, and it declines
    # to state figures it was not given — which matters because the locked
    # hotel/taxi/dinner/per-diem numbers are graded as hallucinations if
    # invented.
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    groq_judge_model: str = "qwen/qwen3.8-27b"

    llm_temperature: float = 0.1

    @staticmethod
    def _usable(value: str) -> bool:
        return value.strip() not in _UNSET_KEYS

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # --- Qdrant ---
    # Absolute (not "./data/...") so the app finds the index no matter which
    # directory it is launched from — NFR-F2 portability, NFR-D4 missing-index
    # messaging only works if we look in the right place.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "atlas_docs"
    qdrant_local_path: str = str(project_root / "data" / "qdrant_index")

    # --- Chunking (FR-B4) ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval (FR-C2) ---
    retrieval_top_k: int = 5

    # --- LangGraph limits (FR-D4, FR-D5) ---
    max_steps: int = 10
    max_tool_calls: int = 3

    # --- Router fallback (FR-D2) ---
    router_confidence_threshold: float = 0.55

    # --- Logging (FR-I) ---
    log_level: str = "INFO"
    run_logs_path: Path = outputs_dir / "run_logs.jsonl"

    # --- Evaluation (FR-J) ---
    eval_cases_path: Path = tests_dir / "eval_cases.jsonl"
    eval_report_path: Path = outputs_dir / "eval_report.json"

    # --- Provider resolution -------------------------------------------
    @property
    def resolved_provider(self) -> str:
        """Which LLM backend the assistant will actually talk to.

        Never raises when no key is set: src.llm_setup builds its client at
        import time and the offline test suite imports that module with no
        credentials at all. With nothing configured we fall back to the
        documented default and let the runtime guard (app.py, evaluate.py)
        produce the actionable "go get a key" message.
        """
        requested = self.llm_provider.strip().lower()
        if requested in ("gemini", "groq"):
            return requested
        if requested != "auto":
            raise ValueError(
                f"LLM_PROVIDER must be 'auto', 'gemini' or 'groq', got {requested!r}"
            )
        if self._usable(self.google_api_key):
            return "gemini"
        if self._usable(self.groq_api_key):
            return "groq"
        return "gemini"

    @property
    def active_api_key(self) -> str:
        return (
            self.groq_api_key
            if self.resolved_provider == "groq"
            else self.google_api_key
        )

    @property
    def active_model(self) -> str:
        return (
            self.groq_model if self.resolved_provider == "groq" else self.gemini_model
        )

    @property
    def active_judge_model(self) -> str:
        if self.resolved_provider == "groq":
            return self.groq_judge_model
        return self.gemini_judge_model

    def has_any_llm_key(self) -> bool:
        """True when at least one provider could actually serve a request."""
        return self._usable(self.google_api_key) or self._usable(self.groq_api_key)


settings = Settings()
