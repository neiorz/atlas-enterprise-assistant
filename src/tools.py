"""
Typed tools (FR-F1).

Each tool has a Pydantic input schema (validated, FR-F2) and a typed Pydantic
return object. Static corpus facts (reimbursement caps, leave types) live in
src/corpus_facts.py — a cap comparison or an ID lookup is then always exact,
never guessed, which directly protects the DeepEval Faithfulness/Hallucination
scores (FR-G4).

FR-F2 (validation errors recover): calling a tool with bad input raises
pydantic.ValidationError. src/agent_nodes.py's tools node catches this, feeds
the error text back to the LLM as a feedback message, and allows exactly one
retry before giving up gracefully — the app never crashes on a malformed call.

FR-F3 (tool selection heuristic, documented): the agent node only calls a tool
when the question needs a lookup/calculation retrieval alone can't do reliably
— a bare policy ID, a "how much will I get reimbursed for X + Y", or "what
leave types are there". A question like "what's the vacation policy?" is
answered from retrieval directly, no tool call.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator

from src.corpus_facts import (
    _CAP_SOURCE,
    _CAPS,
    _LEAVE_TYPES,
    _LEAVE_TYPES_SOURCE,
    ExpenseCategory,
    LeaveType,
    TripType,
)
from src.settings import settings

POLICY_INDEX_PATH = settings.data_dir / "policy_index.json"


# ==========================================================================
# Tool 1 — Policy Lookup by ID
# ==========================================================================

_POLICY_ID_PATTERN = re.compile(r"^[A-Z]{2,5}-[A-Z]{2,5}-\d{3}$")


class PolicyLookupInput(BaseModel):
    policy_id: str = Field(
        ..., description="Policy/document ID, e.g. 'FIN-TRA-008' or 'HR-VAC-001'."
    )

    @field_validator("policy_id")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not _POLICY_ID_PATTERN.match(v):
            raise ValueError(
                f"'{v}' is not a valid policy ID. Expected format like 'FIN-TRA-008': "
                f"2-5 uppercase letters, dash, 2-5 uppercase letters, dash, 3 digits."
            )
        return v


class PolicyLookupResult(BaseModel):
    found: bool
    policy_id: str
    domain: str | None = None
    source_filename: str | None = None
    message: str


def _load_policy_index() -> dict[str, dict]:
    if not POLICY_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{POLICY_INDEX_PATH} not found — run `python -m src.ingest` first "
            f"to build the policy ID index."
        )
    return json.loads(POLICY_INDEX_PATH.read_text(encoding="utf-8"))


def policy_lookup(payload: PolicyLookupInput) -> PolicyLookupResult:
    """Map a policy ID to the document that defines it, via the pre-built index."""
    index = _load_policy_index()
    entry = index.get(payload.policy_id)
    if entry is None:
        return PolicyLookupResult(
            found=False,
            policy_id=payload.policy_id,
            message=f"No document in the corpus is registered under policy ID '{payload.policy_id}'.",
        )
    return PolicyLookupResult(
        found=True,
        policy_id=payload.policy_id,
        domain=entry["domain"],
        source_filename=entry["source_filename"],
        message=f"Policy '{payload.policy_id}' is defined in {entry['source_filename']} ({entry['domain']}).",
    )


# ==========================================================================
# Tool 2 — Reimbursement Calculator
# ==========================================================================


class ExpenseItem(BaseModel):
    category: ExpenseCategory
    amount_per_unit: float = Field(
        ...,
        gt=0,
        description="Claimed amount per unit, in the category's native currency.",
    )
    quantity: int = Field(
        default=1,
        gt=0,
        description="Nights (hotel), legs (taxi), persons (client_dinner), days (per_diem).",
    )


class ReimbursementCalculatorInput(BaseModel):
    trip_type: TripType
    items: list[ExpenseItem] = Field(..., min_length=1)


class ExpenseLineResult(BaseModel):
    category: ExpenseCategory
    currency: str
    cap_per_unit: float
    claimed_per_unit: float
    quantity: int
    claimed_total: float
    reimbursable_total: float
    over_cap: bool
    source_filename: str


class ReimbursementCalculatorResult(BaseModel):
    trip_type: TripType
    lines: list[ExpenseLineResult]
    totals_by_currency: dict[str, float]
    any_item_over_cap: bool


def reimbursement_calculator(
    payload: ReimbursementCalculatorInput,
) -> ReimbursementCalculatorResult:
    """Clamp each claimed item to its cap and total by currency.

    Caps come from src/corpus_facts.py (locked corpus facts), so the arithmetic
    is exact and every line cites the document the cap is defined in.
    """
    lines: list[ExpenseLineResult] = []
    totals: dict[str, float] = {}

    for item in payload.items:
        cap_per_unit, currency = _CAPS[item.category][payload.trip_type]
        reimbursable_per_unit = min(item.amount_per_unit, cap_per_unit)
        claimed_total = item.amount_per_unit * item.quantity
        reimbursable_total = reimbursable_per_unit * item.quantity

        lines.append(
            ExpenseLineResult(
                category=item.category,
                currency=currency,
                cap_per_unit=cap_per_unit,
                claimed_per_unit=item.amount_per_unit,
                quantity=item.quantity,
                claimed_total=claimed_total,
                reimbursable_total=reimbursable_total,
                over_cap=item.amount_per_unit > cap_per_unit,
                source_filename=_CAP_SOURCE[item.category],
            )
        )
        totals[currency] = totals.get(currency, 0.0) + reimbursable_total

    return ReimbursementCalculatorResult(
        trip_type=payload.trip_type,
        lines=lines,
        totals_by_currency={k: round(v, 2) for k, v in totals.items()},
        any_item_over_cap=any(line.over_cap for line in lines),
    )


# ==========================================================================
# Tool 3 — List Leave Types
# ==========================================================================


class ListLeaveTypesInput(BaseModel):
    paid_only: bool = Field(
        default=False, description="If true, return only paid leave types."
    )


class ListLeaveTypesResult(BaseModel):
    leave_types: list[LeaveType]
    source_filename: str


def list_leave_types(payload: ListLeaveTypesInput) -> ListLeaveTypesResult:
    """Return the leave catalogue (optionally paid-only) with its source document."""
    types = [t for t in _LEAVE_TYPES if (t.paid or not payload.paid_only)]
    return ListLeaveTypesResult(leave_types=types, source_filename=_LEAVE_TYPES_SOURCE)


# --------------------------------------------------------------------------
# Registry — used by src/llm_setup.py to bind these to the LLM as callable
# tools, and by src/agent_nodes.py to execute them with Pydantic validation.
# --------------------------------------------------------------------------

TOOL_REGISTRY = {
    "policy_lookup": (PolicyLookupInput, policy_lookup),
    "reimbursement_calculator": (
        ReimbursementCalculatorInput,
        reimbursement_calculator,
    ),
    "list_leave_types": (ListLeaveTypesInput, list_leave_types),
}
