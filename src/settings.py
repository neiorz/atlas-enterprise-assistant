"""
Centralized configuration for the Atlas Enterprise Assistant.

"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- LLM (Google Gemini) ---
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_judge_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1

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


settings = Settings()
