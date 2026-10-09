"""End-to-end smoke test of the brief's three-turn demo scenario.

Why this exists: tests/evaluate.py gives every gold case a FRESH session so
memory can't leak between cases — which means it can never prove FR-E
(session memory works). This script runs the exact demo from the brief's
"Real Case Scenario" in ONE session:

  Turn 1 (EN)        reimbursement question -> finance, FIN-001 + FIN-002,
                     one tool call, cited answer
  Turn 2 (follow-up) "hotel cap domestic?"  -> must resolve using turn 1's
                     context via session memory (FR-E1)
  Turn 3 (AR)        Arabic client-dinner cap -> finance, FIN-009, RTL answer
                     in Arabic (FR-G3, FR-H3)

Usage:
    python scripts/smoke_demo.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.settings import settings

TURNS = [
    (
        "EN / finance / multi-doc + tool",
        (
            "I'm a new joiner. I traveled to a client meeting last week and paid for "
            "a taxi and a hotel. How do I submit the reimbursement, and how many days "
            "do I have to do it?"
        ),
    ),
    (
        "EN / finance / follow-up (MEMORY)",
        "And what's the cap on the hotel for a domestic trip?",
    ),
    (
        "AR / finance / cross-language",
        "كم الحد الأقصى لعشاء العملاء لكل شخص؟",
    ),
]

# Filenames the answer should be grounded in, per the brief's demo target.
EXPECTED_HINTS = [
    ("FIN-001", "FIN-002"),
    ("FIN-001",),
    ("FIN-009",),
]


def _looks_cited(answer: str, hints: tuple[str, ...]) -> bool:
    return any(h in answer for h in hints)


def main() -> int:
    if not settings.has_any_llm_key():
        print(
            "ERROR: no LLM API key in .env. Set GROQ_API_KEY "
            "(https://console.groq.com/keys) or GOOGLE_API_KEY "
            "(https://aistudio.google.com/apikey) — both are free.",
            file=sys.stderr,
        )
        return 1

    from src.graph import run_turn

    session_id = str(uuid.uuid4())  # ONE session for all three turns (FR-E)
    print(f"session_id = {session_id}\n")

    failures: list[str] = []

    for i, ((label, question), hints) in enumerate(zip(TURNS, EXPECTED_HINTS), 1):
        print(f"--- Turn {i}: {label}")
        print(f"    Q: {question}")
        started = time.perf_counter()
        state = run_turn(question, session_id)
        elapsed = time.perf_counter() - started

        answer = state.get("answer", "")
        domain = state.get("domain")
        sources = state.get("sources", [])
        tools = [tc["tool"] for tc in state.get("tool_calls", [])]
        run_id = state.get("run_id", "")

        print(f"    domain={domain}  sources={sources}  tools={tools}  {elapsed:.1f}s")
        print(f"    run_id={run_id}")
        print(f"    A: {answer[:400]}")

        # --- checks ---
        if domain not in ("finance", "multi"):
            failures.append(f"turn {i}: routed to '{domain}', expected finance")
        if not sources:
            failures.append(f"turn {i}: no sources retrieved")
        if not _looks_cited(answer, hints):
            failures.append(
                f"turn {i}: answer doesn't cite any of {hints} (FR-G1/FR-G2)"
            )
        if i == 3 and answer.strip() and not _is_mostly_arabic(answer):
            failures.append("turn 3: Arabic question got a non-Arabic answer (FR-G3)")
        if elapsed > 8:
            failures.append(f"turn {i}: took {elapsed:.1f}s > NFR-B1's 8s budget")
        print()

    # FR-I3: every event of this run must be greppable by run_id
    if settings.run_logs_path.exists():
        logged = settings.run_logs_path.read_text(encoding="utf-8").count(
            f'"session_id": "{session_id}"'
        )
        print(f"run log events for this session: {logged}")
        if logged < 4:
            failures.append(f"only {logged} log events found for the session (FR-I2)")
    else:
        failures.append("outputs/run_logs.jsonl missing (FR-I2)")

    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("  ✓ all demo checks passed (routing, citations, memory, Arabic, latency)")
    return 0


def _is_mostly_arabic(text: str) -> bool:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return arabic > 0 and arabic >= latin * 0.4


if __name__ == "__main__":
    raise SystemExit(main())
