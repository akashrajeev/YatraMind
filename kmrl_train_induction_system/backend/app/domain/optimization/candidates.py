"""Candidate preparation for the optimization pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.domain.optimization.constraints import validate_trainset_safety
from app.domain.optimization.types import ConstraintViolation


@dataclass(frozen=True)
class CandidatePool:
    eligible: List[Dict[str, Any]]
    blocked: List[Dict[str, Any]]
    violations: Dict[str, List[ConstraintViolation]]


def build_candidate_pool(trainsets: Sequence[Dict[str, Any]]) -> CandidatePool:
    """Partition trainsets into eligible and safety-blocked candidates.

    The input/output shape remains legacy-dict compatible so the current
    optimizer can be migrated without changing its public contract.
    """
    eligible: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    violations: Dict[str, List[ConstraintViolation]] = {}

    for trainset in trainsets:
        trainset_id = str(trainset.get("trainset_id", "UNKNOWN"))
        found = validate_trainset_safety(trainset)
        violations[trainset_id] = found
        if any(item.is_blocking for item in found):
            blocked.append(dict(trainset))
        else:
            eligible.append(dict(trainset))

    return CandidatePool(eligible=eligible, blocked=blocked, violations=violations)
