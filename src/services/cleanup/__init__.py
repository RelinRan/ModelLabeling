"""Dataset cleanup: a read-only scan, then an opt-in repair.

Deliberately Qt-free so rules can be unit-tested without a QApplication.
"""
from .fixer import FixPolicy, FixReport, Repair, available_fixes, fix_dataset, plan_repairs
from .report import Issue, Plan, Severity, format_issue
from .runner import scan_dataset

__all__ = [
    "FixPolicy",
    "FixReport",
    "Issue",
    "Plan",
    "Repair",
    "Severity",
    "available_fixes",
    "fix_dataset",
    "format_issue",
    "plan_repairs",
    "scan_dataset",
]
