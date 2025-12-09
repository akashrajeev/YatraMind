# backend/app/ml/multi_depot/feedback_loop.py
"""
ML Feedback Loop / Online Learning
After every simulated or production day, log outcomes and retrain models
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from app.ml.multi_depot.failure_risk_model import FailureRiskPredictor
from app.ml.multi_depot.demand_forecaster import DemandForecaster
from app.ml.multi_depot.service_selection_model import ServiceSelector
from app.ml.multi_depot.rl_stabling_agent import RLStablingAgent
from app.ml.multi_depot.rl_shunting_sequencer import RLShuntingSequencer
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class MultiDepotFeedbackLoop:
    """Feedback loop for continuous learning"""
    
    def __init__(self):
        self.failure_predictor = FailureRiskPredictor()
        self.demand_forecaster = DemandForecaster()
        self.service_selector = ServiceSelector()
    
    async def log_production_outcomes(self, day_date: str, depot_id: str,
                                     outcomes: Dict[str, Any]):
        """
        Log production day outcomes
        
        outcomes: Dict with keys:
        - actual_failures: List of {train_id, failure_time, component, downtime_hours}
        - realized_shunting_time: float (minutes)
        - rollout_delays: List of {train_id, delay_minutes}
        - branding_exposure_delivered: float (hours)
        - decisions: List of decisions made
        - allocations: List of allocations
        """
        try:
            collection = await cloud_db_manager.get_collection("ml_feedback_logs")
            
            feedback_record = {
                "day_date": day_date,
                "depot_id": depot_id,
                "timestamp": datetime.now().isoformat(),
                "outcomes": outcomes,
                "processed": False,
            }
            
            await collection.insert_one(feedback_record)
            logger.info(f"Logged production outcomes for {day_date} at {depot_id}")
            
        except Exception as e:
            logger.error(f"Error logging production outcomes: {e}")
    
    async def process_feedback_and_retrain(self, days_back: int = 7,
                                         incremental: bool = True) -> Dict[str, Any]:
        """
        Process feedback logs and retrain models
        
        Returns:
        - Status and training results for each model
        """
        try:
            # Load feedback logs
            collection = await cloud_db_manager.get_collection("ml_feedback_logs")
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            cursor = collection.find({
                "timestamp": {"$gte": cutoff_date},
                "processed": False,
            })
            
            feedback_logs = await cursor.to_list(None)
            
            if not feedback_logs:
                return {"status": "no_new_data"}
            
            logger.info(f"Processing {len(feedback_logs)} feedback logs")
            
            results = {}
            
            # 1. Retrain Failure Risk Model
            try:
                training_data = await self._prepare_failure_training_data(feedback_logs)
                if training_data:
                    result = await self.failure_predictor.train(training_data, epochs=20)
                    results["failure_risk"] = result
            except Exception as e:
                logger.error(f"Error retraining failure risk model: {e}")
                results["failure_risk"] = {"error": str(e)}
            
            # 2. Retrain Demand Forecaster
            try:
                training_data = await self._prepare_demand_training_data(feedback_logs)
                if training_data:
                    result = await self.demand_forecaster.train(training_data)
                    results["demand_forecast"] = result
            except Exception as e:
                logger.error(f"Error retraining demand forecaster: {e}")
                results["demand_forecast"] = {"error": str(e)}
            
            # 3. Retrain Service Selector
            try:
                training_data = await self._prepare_service_training_data(feedback_logs)
                if training_data:
                    result = await self.service_selector.train(training_data, epochs=20)
                    results["service_selection"] = result
            except Exception as e:
                logger.error(f"Error retraining service selector: {e}")
                results["service_selection"] = {"error": str(e)}
            
            # Mark logs as processed
            await collection.update_many(
                {"_id": {"$in": [log["_id"] for log in feedback_logs]}},
                {"$set": {"processed": True}}
            )
            
            return {"status": "ok", "results": results}
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _prepare_failure_training_data(self, feedback_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare training data for failure risk model"""
        training_data = []
        
        for log in feedback_logs:
            outcomes = log.get("outcomes", {})
            failures = outcomes.get("actual_failures", [])
            decisions = outcomes.get("decisions", [])
            
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                
                train_id = decision.get("train_id")
                if not train_id:
                    continue
                
                # Check if failure occurred
                failure = next((f for f in failures 
                               if isinstance(f, dict) and f.get("train_id") == train_id), None)
                
                if failure:
                    try:
                        failure_time = datetime.fromisoformat(failure.get("failure_time", ""))
                        decision_time = datetime.fromisoformat(log.get("day_date", datetime.now().isoformat()))
                        hours_until_failure = (failure_time - decision_time).total_seconds() / 3600
                        
                        label_24h = 1.0 if 0 < hours_until_failure <= 24 else 0.0
                        label_72h = 1.0 if 0 < hours_until_failure <= 72 else 0.0
                    except:
                        label_24h = 0.0
                        label_72h = 0.0
                else:
                    label_24h = 0.0
                    label_72h = 0.0
                
                training_data.append({
                    "train_id": train_id,
                    "sequences": decision.get("historical_sequences", []),
                    "label_24h": label_24h,
                    "label_72h": label_72h,
                    "component_labels": failure.get("component_risks", [0.0] * 5) if failure else [0.0] * 5,
                })
        
        return training_data
    
    async def _prepare_demand_training_data(self, feedback_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare training data for demand forecaster"""
        training_data = []
        
        for log in feedback_logs:
            day_date = log.get("day_date")
            outcomes = log.get("outcomes", {})
            
            # Actual service trains used
            decisions = outcomes.get("decisions", [])
            actual_service_trains = sum(1 for d in decisions 
                                      if isinstance(d, dict) and d.get("decision") == "SERVICE")
            
            training_data.append({
                "date": day_date,
                "service_trains": actual_service_trains,
                "historical_demand": [],  # Would need to fetch historical
            })
        
        return training_data
    
    async def _prepare_service_training_data(self, feedback_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare training data for service selector"""
        training_data = []
        
        for log in feedback_logs:
            outcomes = log.get("outcomes", {})
            decisions = outcomes.get("decisions", [])
            failures = outcomes.get("actual_failures", [])
            delays = outcomes.get("rollout_delays", [])
            
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                
                train_id = decision.get("train_id")
                decision_type = decision.get("decision", "")
                
                # Service score based on outcomes
                if decision_type == "SERVICE":
                    # Check if train performed well
                    had_failure = any(f.get("train_id") == train_id for f in failures if isinstance(f, dict))
                    had_delay = any(d.get("train_id") == train_id for d in delays if isinstance(d, dict))
                    
                    if not had_failure and not had_delay:
                        service_score = 0.9  # Good performance
                    elif had_delay:
                        service_score = 0.6  # Moderate
                    else:
                        service_score = 0.3  # Poor (failure)
                    
                    selected = 1.0
                else:
                    service_score = 0.2
                    selected = 0.0
                
                training_data.append({
                    "fleet_features": decision.get("fleet_features"),
                    "risk_prediction": decision.get("risk_prediction"),
                    "service_score": service_score,
                    "selected": selected,
                })
        
        return training_data


