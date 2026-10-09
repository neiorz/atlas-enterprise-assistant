"""
Structured run logging (FR-I).

"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from src.settings import settings

logging.basicConfig(level=settings.log_level)
_logger = logging.getLogger("atlas")


def new_run_id() -> str:
    return str(uuid.uuid4())


def log_event(run_id: str, session_id: str, event: str, **fields) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "session_id": session_id,
        "event": event,
        **fields,
    }
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.run_logs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _logger.info("event=%s run_id=%s", event, run_id)
