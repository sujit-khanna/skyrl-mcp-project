"""Acceptance evaluation utilities for analysis requirements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .dsl import check_condition


@dataclass
class AcceptanceResult:
    passed: bool
    failed_conditions: List[str]


def evaluate_accept_if(conditions: Sequence[str] | None, state: Mapping[str, object]) -> AcceptanceResult:
    failed: List[str] = []
    for cond in conditions or []:
        if not check_condition(cond, state):
            failed.append(cond)
    return AcceptanceResult(passed=not failed, failed_conditions=failed)


__all__ = ["AcceptanceResult", "evaluate_accept_if"]
