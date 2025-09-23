# backend/app/services/rule_engine.py
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DurableRulesEngine:
    """Rule-based constraint engine (Drools + PyKE equivalent)"""
    
    def __init__(self):
        self.constraint_rules = {
            "fitness_certificate_expiry": self._check_certificate_expiry,
            "critical_job_cards": self._check_critical_job_cards,
            "mileage_limits": self._check_mileage_limits,
            "maintenance_schedule": self._check_maintenance_schedule,
            "branding_contracts": self._check_branding_contracts
        }
    
    async def apply_constraints(self, trainsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all constraint rules to filter trainsets"""
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
        """Check constraints for a single trainset and return violations"""
        violations = []
        
        for rule_name, rule_func in self.constraint_rules.items():
            try:
                violation = await rule_func(trainset)
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.error(f"Rule {rule_name} failed: {e}")
                violations.append(f"Rule {rule_name} validation error")
        
        return violations
    
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
        
        return None
    
    async def _check_branding_contracts(self, trainset: Dict[str, Any]) -> str:
        """Check branding contract constraints"""
        branding = trainset.get("branding", {})
        
        if branding.get("current_advertiser") != "None":
            # Check if contract is expiring soon
            # This would parse the contract_expiry date and check
            return None  # Placeholder - would implement date checking
        
        return None
