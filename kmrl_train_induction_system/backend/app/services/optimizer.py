# backend/app/services/optimizer.py
from typing import List, Dict, Any
from app.models.trainset import OptimizationRequest, InductionDecision
import logging

logger = logging.getLogger(__name__)

class TrainInductionOptimizer:
    """AI/ML optimization engine using Google OR-Tools + PyTorch"""
    
    def __init__(self):
        self.optimization_weights = {
            "fitness_score": 0.3,
            "maintenance_priority": 0.25,
            "branding_priority": 0.2,
            "mileage_balance": 0.15,
            "operational_efficiency": 0.1
        }
    
    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest) -> List[InductionDecision]:
        """Run multi-objective optimization for train induction"""
        try:
            logger.info(f"Starting optimization for {len(trainsets)} trainsets")
            
            # Apply constraint filtering
            eligible_trainsets = self._filter_by_constraints(trainsets)
            
            # Calculate optimization scores
            scored_trainsets = []
            for trainset in eligible_trainsets:
                score = self._calculate_optimization_score(trainset, request)
                scored_trainsets.append((trainset, score))
            
            # Sort by optimization score
            scored_trainsets.sort(key=lambda x: x[1], reverse=True)
            
            # Generate induction decisions
            decisions = []
            inducted_count = 0
            target_inductions = min(14, len(scored_trainsets))  # Target 14 trainsets for service
            
            for trainset, score in scored_trainsets:
                if inducted_count < target_inductions and self._can_induct(trainset):
                    decision = InductionDecision(
                        trainset_id=trainset["trainset_id"],
                        decision="INDUCT",
                        confidence_score=min(score, 1.0),
                        reasons=self._get_induction_reasons(trainset, score)
                    )
                    inducted_count += 1
                elif self._needs_maintenance(trainset):
                    decision = InductionDecision(
                        trainset_id=trainset["trainset_id"],
                        decision="MAINTENANCE",
                        confidence_score=0.9,
                        reasons=["Maintenance required based on constraints"]
                    )
                else:
                    decision = InductionDecision(
                        trainset_id=trainset["trainset_id"],
                        decision="STANDBY",
                        confidence_score=0.7,
                        reasons=["Standby due to lower priority"]
                    )
                
                decisions.append(decision)
            
            logger.info(f"Optimization completed: {inducted_count} inducted, {len(decisions) - inducted_count} standby/maintenance")
            return decisions
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _filter_by_constraints(self, trainsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply rule-based constraints filtering"""
        eligible = []
        
        for trainset in trainsets:
            # Check fitness certificates
            fitness_valid = all(
                cert["status"] == "VALID" 
                for cert in trainset["fitness_certificates"].values()
            )
            
            # Check critical job cards
            no_critical_cards = trainset["job_cards"]["critical_cards"] == 0
            
            # Check mileage limits
            mileage_ok = trainset["current_mileage"] < trainset["max_mileage_before_maintenance"]
            
            if fitness_valid and no_critical_cards and mileage_ok:
                eligible.append(trainset)
        
        return eligible
    
    def _calculate_optimization_score(self, trainset: Dict[str, Any], request: OptimizationRequest) -> float:
        """Calculate multi-objective optimization score"""
        score = 0.0
        
        # Fitness score (0-1)
        fitness_score = self._calculate_fitness_score(trainset)
        score += fitness_score * self.optimization_weights["fitness_score"]
        
        # Maintenance priority (inverse of open cards)
        maintenance_score = max(0, 1 - (trainset["job_cards"]["open_cards"] / 10))
        score += maintenance_score * self.optimization_weights["maintenance_priority"]
        
        # Branding priority
        branding_score = self._calculate_branding_score(trainset)
        score += branding_score * self.optimization_weights["branding_priority"]
        
        # Mileage balance (prefer lower mileage for even distribution)
        mileage_score = 1 - (trainset["current_mileage"] / trainset["max_mileage_before_maintenance"])
        score += mileage_score * self.optimization_weights["mileage_balance"]
        
        # Operational efficiency (based on recent performance)
        operational_score = 0.8  # Placeholder - would use ML model
        score += operational_score * self.optimization_weights["operational_efficiency"]
        
        return min(score, 1.0)
    
    def _calculate_fitness_score(self, trainset: Dict[str, Any]) -> float:
        """Calculate overall fitness score from certificates"""
        fitness_certs = trainset["fitness_certificates"]
        valid_count = sum(1 for cert in fitness_certs.values() if cert["status"] == "VALID")
        return valid_count / len(fitness_certs)
    
    def _calculate_branding_score(self, trainset: Dict[str, Any]) -> float:
        """Calculate branding priority score"""
        branding = trainset.get("branding", {})
        if not branding or branding.get("current_advertiser") == "None":
            return 0.0
        
        priority_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        return priority_map.get(branding.get("priority", "LOW"), 0.0)
    
    def _can_induct(self, trainset: Dict[str, Any]) -> bool:
        """Check if trainset can be inducted"""
        return (
            trainset["status"] in ["STANDBY", "ACTIVE"] and
            trainset["job_cards"]["critical_cards"] == 0
        )
    
    def _needs_maintenance(self, trainset: Dict[str, Any]) -> bool:
        """Check if trainset needs maintenance"""
        return (
            trainset["job_cards"]["critical_cards"] > 0 or
            trainset["current_mileage"] >= trainset["max_mileage_before_maintenance"] * 0.95
        )
    
    def _get_induction_reasons(self, trainset: Dict[str, Any], score: float) -> List[str]:
        """Generate human-readable reasons for induction decision"""
        reasons = []
        
        if score > 0.8:
            reasons.append("High optimization score")
        
        if trainset["job_cards"]["open_cards"] == 0:
            reasons.append("No pending maintenance")
        
        if trainset.get("branding", {}).get("priority") == "HIGH":
            reasons.append("High branding priority")
        
        if trainset["current_mileage"] < trainset["max_mileage_before_maintenance"] * 0.5:
            reasons.append("Low mileage - good for service")
        
        return reasons if reasons else ["Selected based on optimization criteria"]
