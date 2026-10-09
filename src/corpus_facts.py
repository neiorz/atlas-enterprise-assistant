"""Locked corpus facts — the single source of truth for tool arithmetic (FR-G4).

Every number here is explicitly stated in the corpus (see
data/CORPUS_README.md, "Locked Facts"). They live in their own module so:

1. tools.py stays focused on schemas + logic, comfortably under NFR-C1's
   ~250-line guideline.
2. There is exactly one place to check when a grader asks "where does the
   1,500 EGP cap come from?" — never the LLM's memory.

If a fact here ever disagrees with the corpus, the DeepEval Faithfulness and
Hallucination metrics will fail the system, not the model's recall.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ExpenseCategory = Literal["hotel", "taxi", "client_dinner", "per_diem"]
TripType = Literal["domestic", "international"]

# Per-unit caps by category and trip type, in the category's native currency.
_CAPS: dict[ExpenseCategory, dict[TripType, tuple[float, str]]] = {
    "hotel": {"domestic": (1500.0, "EGP"), "international": (200.0, "USD")},
    "per_diem": {"domestic": (500.0, "EGP"), "international": (75.0, "USD")},
    "taxi": {"domestic": (800.0, "EGP"), "international": (800.0, "EGP")},
    "client_dinner": {"domestic": (2000.0, "EGP"), "international": (2000.0, "EGP")},
}

# Which document each cap is defined in, so the calculator can cite its source.
_CAP_SOURCE: dict[ExpenseCategory, str] = {
    "hotel": "FIN-001-travel-reimbursement-policy.md",
    "per_diem": "FIN-004-per-diem-rates.md",
    "taxi": "FIN-008-بدل-المواصلات.pdf",
    "client_dinner": "FIN-009-عشاء-العملاء.docx",
}


class LeaveType(BaseModel):
    """One entry of Atlas Industries' leave catalogue."""

    name: str
    paid: bool
    entitlement: str


_LEAVE_TYPES: list[LeaveType] = [
    LeaveType(
        name="Annual Vacation",
        paid=True,
        entitlement="15 days for new hires; up to 25 days after 10+ years; up to 5 days carry-over.",
    ),
    LeaveType(
        name="Sick Leave",
        paid=True,
        entitlement="Up to 14 paid days/year; medical certificate required for 3+ consecutive days.",
    ),
    LeaveType(
        name="Maternity Leave",
        paid=True,
        entitlement="4 months fully paid; additional 2 months unpaid on request.",
    ),
    LeaveType(
        name="Paternity Leave",
        paid=True,
        entitlement="10 working days fully paid, within 3 months of the birth.",
    ),
    LeaveType(
        name="Bereavement Leave",
        paid=True,
        entitlement="5 working days for immediate family; 2 days for in-laws/grandparents/grandchildren.",
    ),
    LeaveType(
        name="Marriage Leave",
        paid=True,
        entitlement="5 consecutive working days, once per employment tenure.",
    ),
    LeaveType(
        name="Hajj / Religious Pilgrimage Leave",
        paid=True,
        entitlement="Up to 21 calendar days, once per tenure; 60 days' notice required.",
    ),
    LeaveType(
        name="Examination Leave",
        paid=True,
        entitlement="Up to 5 working days/year for accredited education programs.",
    ),
    LeaveType(
        name="Jury / Civic Duty Leave",
        paid=True,
        entitlement="As required by court summons, with documentation.",
    ),
    LeaveType(
        name="Personal Unpaid Leave",
        paid=False,
        entitlement="Up to 30 calendar days/year, manager approval required.",
    ),
    LeaveType(
        name="Sabbatical",
        paid=False,
        entitlement="After 5 years of service; up to 3 months unpaid, role reinstated.",
    ),
    LeaveType(
        name="Study Leave (Long)",
        paid=False,
        entitlement="Up to 12 months unpaid for degree programs; CEO-level approval.",
    ),
]

_LEAVE_TYPES_SOURCE = "HR-006-leave-types-guide.docx"
