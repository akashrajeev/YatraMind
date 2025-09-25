# backend/app/services/solver.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from ortools.sat.python import cp_model


@dataclass
class SolverWeights:
    readiness: float = 0.35
    reliability: float = 0.30
    branding: float = 0.20
    shunt: float = 0.10
    mileage_balance: float = 0.05


class RoleAssignmentSolver:
    """Builds and solves a CP-SAT model for rake role assignment.

    Decision variables:
      x[r, role] in {0,1}, role in {service, standby, IBL}
      Optional: y[r, bay] if stabling used (not fully implemented here)
    """

    def __init__(self, *, required_service_count: int, weights: SolverWeights | None = None):
        self.required_service_count = required_service_count
        self.weights = weights or SolverWeights()

    def solve(self, features: List[Dict[str, Any]], *, bays: List[str] | None = None, capacities: Dict[str, int] | None = None) -> Dict[str, Any]:
        roles = ["service", "standby", "IBL"]
        model = cp_model.CpModel()

        # Variables
        x: Dict[Tuple[int, str], cp_model.IntVar] = {}
        for i, _ in enumerate(features):
            for role in roles:
                x[(i, role)] = model.NewBoolVar(f"x_{i}_{role}")

        # Each rake exactly one role
        for i, _ in enumerate(features):
            model.Add(sum(x[(i, role)] for role in roles) == 1)

        # Required service count
        model.Add(sum(x[(i, "service")] for i in range(len(features))) == self.required_service_count)

        # Rules engine enforced flags inside features (booleans)
        for i, f in enumerate(features):
            # If not allowed in service, force x_service == 0
            if not f.get("allowed_service", True):
                model.Add(x[(i, "service")] == 0)
            # If must IBL, force IBL
            if f.get("must_ibl", False):
                model.Add(x[(i, "IBL")] == 1)

        # Optional bay capacities (cleaning bay, stabling) - simplified aggregate
        if capacities:
            if "cleaning" in capacities:
                # Only rakes with cleaning_available can be service; others cannot
                for i, f in enumerate(features):
                    if not f.get("cleaning_available", False):
                        model.Add(x[(i, "service")] == 0)

        # Objective: weighted sum
        # Normalize components to scaled ints for CP-SAT
        scale = 1000
        terms = []
        for i, f in enumerate(features):
            readiness = int(scale * float(f.get("readiness", 0.0)))
            reliability = int(scale * float(f.get("reliability", 0.0)))
            branding = int(scale * float(f.get("branding", 0.0)))
            shunt_cost = int(scale * float(1.0 - f.get("shunt_cost_norm", 0.0)))  # lower cost is better
            # For mileage balance, we minimize variance indirectly by preferring lower recent km
            mileage_bal = int(scale * float(1.0 - f.get("km_30d_norm", 0.0)))

            score_i = (
                int(self.weights.readiness * readiness)
                + int(self.weights.reliability * reliability)
                + int(self.weights.branding * branding)
                + int(self.weights.shunt * shunt_cost)
                + int(self.weights.mileage_balance * mileage_bal)
            )
            if score_i != 0:
                terms.append(score_i * x[(i, "service")])

        model.Maximize(sum(terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 15.0
        status = solver.Solve(model)

        assignments: List[Dict[str, Any]] = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i, f in enumerate(features):
                role = None
                for r in roles:
                    if solver.Value(x[(i, r)]) == 1:
                        role = r
                        break
                assignments.append({
                    "trainset_id": f.get("trainset_id"),
                    "role": role or "standby",
                    "objective_contrib": float(solver.ObjectiveValue()) if role == "service" else 0.0,
                })

        return {
            "status": "ok" if assignments else "infeasible",
            "assignments": assignments,
            "objective": float(solver.ObjectiveValue()) if assignments else 0.0,
        }


