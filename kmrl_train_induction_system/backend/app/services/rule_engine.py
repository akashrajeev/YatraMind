# backend/app/services/rule_engine.py
from typing import List, Dict, Any
import logging
from datetime import datetime
import requests
from app.config import settings

try:
    # durable_rules is optional; we fallback to pure-Python checks if unavailable
    from durable_rules import ruleset, when_all, m, c, post
    _HAS_DURABLE = True
except Exception:
    _HAS_DURABLE = False

logger = logging.getLogger(__name__)

class DurableRulesEngine:
    """Rule-based constraint engine (Drools + PyKE equivalent)

    Implementation details:
    - If durable_rules is available, constraints are evaluated via a small ruleset and
      violations are collected from rule actions.
    - If durable_rules is not present, a safe Python fallback is used.
    """
    
    def __init__(self):
        self.constraint_rules = {
            "fitness_certificate_expiry": self._check_certificate_expiry,
            "critical_job_cards": self._check_critical_job_cards,
            "mileage_limits": self._check_mileage_limits,
            "maintenance_schedule": self._check_maintenance_schedule,
            "branding_contracts": self._check_branding_contracts
        }
    
    async def apply_constraints(self, trainsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all constraint rules to filter trainsets."""
        try:
            logger.info(f"Applying constraints to {len(trainsets)} trainsets")
            valid_trainsets = []
            for trainset in trainsets:
                if await self._validate_trainset(trainset):
                    valid_trainsets.append(trainset)
            logger.info(f"Constraint filtering: {len(valid_trainsets)}/{len(trainsets)} trainsets valid")
            return valid_trainsets
        except Exception as e:
            logger.error(f"Constraint application failed: {e}")
            raise
    
    async def check_constraints(self, trainset: Dict[str, Any]) -> List[str]:
        """Check constraints for a single trainset and return violations."""
        if _HAS_DURABLE:
            try:
                return self._check_with_durable(trainset)
            except Exception as e:
                logger.warning(f"durable_rules evaluation failed, falling back: {e}")
        # Fallback path
        violations: List[str] = []
        for rule_name, rule_func in self.constraint_rules.items():
            try:
                violation = await rule_func(trainset)
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.error(f"Rule {rule_name} failed: {e}")
                violations.append(f"Rule {rule_name} validation error")
        return violations

    def _check_with_durable(self, trainset: Dict[str, Any]) -> List[str]:
        """Evaluate constraints using durable_rules and return violations."""
        # Ensure rules are defined once
        _ensure_rules_defined()
        
        # Post the fact to the static ruleset
        # We attach a unique 'sid' (session id) or just use the fact payload
        # durable_rules raises exception on violation if handled that way, 
        # but here we are collecting violations in a global or context-local way?
        # The original code used a closure to capture 'violations' list.
        # With a static ruleset, we can't easily capture a local variable from the rule action.
        # WE NEED A WAY TO GET RESULTS BACK.
        
        # durable_rules is tricky with return values. 
        # Standard pattern: The rule modifies the state or posts a new event.
        # OR, we can attach a mutable list to the fact itself? 
        # durable_rules usually works on JSON-serializable dicts.
        
        # Workaround: 
        # Since durable_rules in Python runs in-process, we can cheat slightly if the engine allows objects,
        # but usually it expects dicts.
        # 
        # Alternative: The original code was creating a NEW ruleset every time to capture the 'violations' list via closure.
        # To fix the leak, we must use a static ruleset.
        # To get results, we can use a module-level dictionary mapping request_id to violations?
        # Or, since this is 'durable_rules', maybe we can't use it easily for this synchronous check without side effects?
        
        # Let's look at how we can pass context.
        # If we post {'trainset_id': '...', ...}, the rule triggers.
        # The action function receives 'c'.
        # We need to map back to the request.
        
        # Let's use a simple global registry for the current context if we assume single-threaded or use contextvars?
        # No, this is async, so multiple requests can be in flight.
        
        # Better approach:
        # The 'fact' can contain a 'result_container_id'.
        # We keep a global dictionary: _results = { 'id': [] }
        # The rule appends to _results[c.m.result_container_id].
        
        import uuid
        request_id = str(uuid.uuid4())
        _rule_results[request_id] = []
        
        fact = trainset.copy()
        fact['result_container_id'] = request_id
        
        try:
            post('kmrl_induction_rules', fact)
        except Exception as e:
            # durable_rules might raise error if ruleset not found or other issues
            logger.error(f"Durable rules error: {e}")
            
        violations = _rule_results.pop(request_id, [])
        return violations

# Global storage for rule results (temporary for the request duration)
_rule_results: Dict[str, List[str]] = {}
_rules_defined = False

def _ensure_rules_defined():
    global _rules_defined
    if _rules_defined:
        return
        
    # Define static ruleset
    rs_name = 'kmrl_induction_rules'
    
    try:
        @when_all(rs_name, m.fitness_certificates.matches(True))
        def certs_ok(c):
            pass

        @when_all(rs_name, m.job_cards.matches(True))
        def cards_present(c):
            pass

        @when_all(rs_name, m.fitness_certificates.any_item.status == 'EXPIRED')
        def rule_cert_expired(c):
            _add_violation(c, "A certificate expired")

        @when_all(rs_name, m.fitness_certificates.any_item.status == 'EXPIRING_SOON')
        def rule_cert_expiring(c):
            _add_violation(c, "A certificate expiring soon")

        @when_all(rs_name, m.job_cards.critical_cards > 0)
        def rule_critical_cards(c):
            _add_violation(c, "Critical job cards pending")

        @when_all(rs_name, (m.current_mileage >= m.max_mileage_before_maintenance))
        def rule_mileage_limit(c):
            _add_violation(c, "Mileage limit exceeded")

        @when_all(rs_name, m.status == 'MAINTENANCE')
        def rule_maintenance(c):
            _add_violation(c, "Currently in maintenance")
            
        _rules_defined = True
        logger.info(f"Ruleset '{rs_name}' defined successfully")
        
    except Exception as e:
        logger.error(f"Failed to define ruleset: {e}")

def _add_violation(c, msg):
    # Helper to add violation to the correct result container
    rid = c.m.result_container_id
    if rid in _rule_results:
        _rule_results[rid].append(msg)


class DroolsAdapterEngine:
    """Thin adapter to call a Java Drools REST service if configured.

    It translates MongoDB trainset documents into a Drools-friendly JSON payload,
    posts to the Drools service, and returns violations and suggestions.
    """

    def __init__(self, service_url: str | None = None):
        self.service_url = service_url or settings.drools_service_url

    async def evaluate_assignment(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if not self.service_url:
            return {"allowed": False, "violations": ["Drools service not configured"], "suggested_fix": None}
        payload = self._to_drools_payload(candidate)
        try:
            resp = requests.post(f"{self.service_url.rstrip('/')}/evaluate", json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return {
                "allowed": bool(data.get("allowed", False)),
                "violations": data.get("violations", []),
                "suggested_fix": data.get("suggested_fix"),
            }
        except Exception as e:
            return {"allowed": False, "violations": [f"Drools error: {e}"], "suggested_fix": None}

    def _to_drools_payload(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trainset": {
                "id": candidate.get("trainset_id"),
                "status": candidate.get("status"),
                "mileage": candidate.get("current_mileage"),
                "fitness": candidate.get("fitness_certificates", {}),
                "job_cards": candidate.get("job_cards", {}),
                "branding": candidate.get("branding", {}),
            },
            "assignment": candidate.get("assignment", {}),
            "context": {
                "dawn": datetime.now().isoformat(),
                "cleaning_capacity": candidate.get("cleaning_capacity", {}),
                "stabling": candidate.get("stabling", {}),
                "mileage_bounds": candidate.get("mileage_bounds", {}),
            },
        }
    
    async def _validate_trainset(self, trainset: Dict[str, Any]) -> bool:
        """Validate if trainset meets all constraints"""
        violations = await self.check_constraints(trainset)
        return len(violations) == 0
    
    async def _check_certificate_expiry(self, trainset: Dict[str, Any]) -> str:
        """Check fitness certificate expiry constraints"""
        fitness_certs = trainset["fitness_certificates"]
        
        for cert_type, cert_data in fitness_certs.items():
            if cert_data["status"] == "EXPIRED":
                return f"{cert_type} certificate expired"
            elif cert_data["status"] == "EXPIRING_SOON":
                return f"{cert_type} certificate expiring soon"
        
        return None
    
    async def _check_critical_job_cards(self, trainset: Dict[str, Any]) -> str:
        """Check critical job card constraints"""
        job_cards = trainset["job_cards"]
        
        if job_cards["critical_cards"] > 0:
            return f"{job_cards['critical_cards']} critical job cards pending"
        
        return None
    
    async def _check_mileage_limits(self, trainset: Dict[str, Any]) -> str:
        """Check mileage limit constraints"""
        current_mileage = trainset["current_mileage"]
        max_mileage = trainset["max_mileage_before_maintenance"]
        
        if current_mileage >= max_mileage:
            return f"Mileage limit exceeded: {current_mileage}/{max_mileage} km"
        elif current_mileage >= max_mileage * 0.95:
            return f"Mileage approaching limit: {current_mileage}/{max_mileage} km"
        
        return None
    
    async def _check_maintenance_schedule(self, trainset: Dict[str, Any]) -> str:
        """Check maintenance schedule constraints"""
        # This would integrate with maintenance scheduling system
        # For now, check if trainset is in maintenance status
        if trainset["status"] == "MAINTENANCE":
            return "Currently in maintenance"
        # Withdrawal-required job-card
        if any(bool(j.get("requires_withdrawal", False)) for j in (trainset.get("job_cards_list") or [])):
            return "Job-card requires withdrawal"
        # Cleaning compliance
        if trainset.get("requires_cleaning", False) and not trainset.get("has_cleaning_slot", False):
            return "No cleaning slot before departure"
        return None
    
    async def _check_branding_contracts(self, trainset: Dict[str, Any]) -> str:
        """Check branding contract constraints"""
        branding = trainset.get("branding", {})
        
        if branding.get("current_advertiser") != "None":
            # Check exposure requirement
            try:
                required_hours = float(branding.get("required_hours_next_n", 0))
                n_days = int(branding.get("window_days", 7))
                # Heuristic available hours if in service each day
                planned_hours = float(branding.get("planned_service_hours", 0))
                if planned_hours < required_hours:
                    return "Branding exposure shortfall in window"
            except Exception:
                return None
        
        return None
