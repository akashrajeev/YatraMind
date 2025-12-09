# backend/app/ml/multi_depot/simulation_engine.py
"""
Multi-Depot Simulation Engine (Main Integration)
Ties together all AI modules for end-to-end simulation
"""
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime, timedelta
import uuid

from app.ml.multi_depot.config import DepotConfig, FleetFeatures, SimulationRun, MultiDepotState
from app.ml.multi_depot.failure_risk_model import FailureRiskPredictor
from app.ml.multi_depot.demand_forecaster import DemandForecaster
from app.ml.multi_depot.service_selection_model import ServiceSelector
from app.ml.multi_depot.rl_stabling_agent import RLStablingAgent
from app.ml.multi_depot.rl_shunting_sequencer import RLShuntingSequencer
from app.ml.multi_depot.transfer_decider import TransferDecider
from app.ml.multi_depot.explainability import AIExplainability, SafetyGuard
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class MultiDepotSimulationEngine:
    """Main AI-driven multi-depot simulation engine"""
    
    def __init__(self):
        self.failure_predictor = FailureRiskPredictor()
        self.demand_forecaster = DemandForecaster()
        self.service_selector = ServiceSelector()
        self.transfer_decider = TransferDecider()
        self.explainability = AIExplainability()
        self.safety_guard = SafetyGuard()
        
        # RL agents (per-depot)
        self.stabling_agents: Dict[str, RLStablingAgent] = {}
        self.shunting_sequencers: Dict[str, RLShuntingSequencer] = {}
        
        self._initialized = False
    
    async def initialize(self, depot_configs: List[DepotConfig]):
        """Initialize all AI models and RL agents"""
        if self._initialized:
            return
        
        logger.info("Initializing Multi-Depot Simulation Engine...")
        
        # Load predictive models
        await self.failure_predictor.load_model()
        await self.demand_forecaster.load_model()
        await self.service_selector.load_model()
        await self.transfer_decider.load_model()
        
        # Initialize RL agents per depot
        for depot_config in depot_configs:
            # Stabling agent (shared across depots for multi-depot coordination)
            if not self.stabling_agents:
                agent = RLStablingAgent(depot_configs)
                await agent.load_model()
                self.stabling_agents["multi_depot"] = agent
            
            # Shunting sequencer (per-depot)
            sequencer = RLShuntingSequencer(depot_config.depot_id)
            await sequencer.load_model()
            self.shunting_sequencers[depot_config.depot_id] = sequencer
        
        self._initialized = True
        logger.info("Multi-Depot Simulation Engine initialized")
    
    async def simulate(self, depot_configs: List[DepotConfig],
                      fleet_features_list: List[FleetFeatures],
                      sim_days: int = 1,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full multi-depot simulation
        
        Returns:
        - Complete simulation results with AI decisions, allocations, transfers, schedules
        """
        await self.initialize(depot_configs)
        
        simulation_id = str(uuid.uuid4())
        logger.info(f"Starting multi-depot simulation {simulation_id}")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        results = {
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat(),
            "depot_configs": [dc.dict() for dc in depot_configs],
            "sim_days": sim_days,
            "seed": seed,
            "daily_results": [],
        }
        
        # Run simulation for each day
        for day in range(sim_days):
            day_date = datetime.now() + timedelta(days=day)
            logger.info(f"Simulating day {day + 1}/{sim_days}: {day_date.date()}")
            
            day_result = await self._simulate_day(
                depot_configs, fleet_features_list, day_date
            )
            
            results["daily_results"].append(day_result)
        
        # Aggregate metrics
        results["aggregated_metrics"] = self._aggregate_metrics(results["daily_results"])
        
        # Save simulation run
        await self._save_simulation_run(results)
        
        logger.info(f"Simulation {simulation_id} completed")
        return results
    
    async def _simulate_day(self, depot_configs: List[DepotConfig],
                           fleet_features_list: List[FleetFeatures],
                           date: datetime) -> Dict[str, Any]:
        """Simulate a single day"""
        
        # Step 1: Forecast demand per depot
        logger.info("Forecasting demand...")
        demand_forecasts = {}
        for depot_config in depot_configs:
            forecast = await self.demand_forecaster.forecast(
                date, depot_config.depot_id
            )
            demand_forecasts[depot_config.depot_id] = forecast["required_service_trains"]
        
        # Step 2: Predict failure risks for all trains
        logger.info("Predicting failure risks...")
        risk_predictions = {}
        for fleet_features in fleet_features_list:
            prediction = await self.failure_predictor.predict(fleet_features)
            risk_predictions[fleet_features.train_id] = prediction
        
        # Step 3: Service selection (rank trains)
        logger.info("Selecting trains for service...")
        ranked_trains = await self.service_selector.rank_trains(
            fleet_features_list, risk_predictions,
            required_count=sum(demand_forecasts.values())
        )
        
        # Step 4: Create multi-depot state
        state = self._create_state(depot_configs, fleet_features_list, 
                                  risk_predictions, demand_forecasts)
        
        # Step 5: RL Stabling allocation
        logger.info("Allocating stabling using RL agent...")
        stabling_allocations = []
        stabling_agent = self.stabling_agents.get("multi_depot")
        
        for ranked_train in ranked_trains:
            train_id = ranked_train["train_id"]
            fleet_features = next((f for f in fleet_features_list 
                                 if f.train_id == train_id), None)
            
            if not fleet_features:
                continue
            
            # Select action using RL agent
            if stabling_agent:
                action = stabling_agent.select_action(state, training=False)
            else:
                # Fallback: assign to first depot
                action = {
                    "depot_id": depot_configs[0].depot_id,
                    "depot_name": depot_configs[0].depot_name,
                    "location_type": "bay",
                    "bay_id": 1,
                }
            
            # Validate with safety guard
            is_valid, rejection_reason = self.safety_guard.validate_service_decision(
                fleet_features.dict(), ranked_train.get("selected", False) and "SERVICE" or "STANDBY"
            )
            
            if not is_valid:
                logger.warning(f"Safety guard rejected allocation for {train_id}: {rejection_reason}")
                action["rejected"] = True
                action["rejection_reason"] = rejection_reason
            
            # Generate explanation
            explanation = self.explainability.explain_stabling_allocation(
                action["depot_id"], action["location_type"], action.get("bay_id"),
                {
                    "risk_score": risk_predictions.get(train_id, {}).get("risk_24h", 0.1),
                    "turnout_time": 10.0,  # Placeholder
                    "branding_priority": fleet_features.branding_priority,
                }
            )
            
            stabling_allocations.append({
                "train_id": train_id,
                "allocation": action,
                "explanation": explanation,
                "service_score": ranked_train["service_score"],
            })
        
        # Step 6: Inter-depot transfer decisions
        logger.info("Evaluating inter-depot transfers...")
        transfer_decisions = []
        for i, from_depot in enumerate(depot_configs):
            for to_depot in depot_configs[i+1:]:
                # Evaluate transfers for standby trains
                for fleet_features in fleet_features_list:
                    if fleet_features.current_location.get("depot") == from_depot.depot_id:
                        transfer_decision = await self.transfer_decider.decide_transfer(
                            fleet_features, from_depot, to_depot,
                            demand_forecasts, risk_predictions.get(fleet_features.train_id)
                        )
                        
                        if transfer_decision["should_transfer"]:
                            transfer_decisions.append(transfer_decision)
        
        # Step 7: Shunting schedule optimization (per depot)
        logger.info("Optimizing shunting schedules...")
        shunting_schedules = {}
        for depot_config in depot_configs:
            sequencer = self.shunting_sequencers.get(depot_config.depot_id)
            if sequencer:
                # Get operations for this depot
                depot_operations = [
                    alloc for alloc in stabling_allocations
                    if alloc["allocation"].get("depot_id") == depot_config.depot_id
                ]
                
                # Convert to shunting operations format
                operations = self._convert_to_shunting_operations(depot_operations, depot_config)
                
                # Optimize sequence
                optimized = sequencer.optimize_sequence(operations, available_window_min=120)
                shunting_schedules[depot_config.depot_id] = optimized
        
        # Step 8: Generate explanations
        explanations = {}
        for ranked_train in ranked_trains[:10]:  # Top 10
            train_id = ranked_train["train_id"]
            fleet_features = next((f for f in fleet_features_list 
                                 if f.train_id == train_id), None)
            
            if fleet_features and self.service_selector.model:
                features = ranked_train["features"]
                feature_names = self.service_selector.feature_names
                explanation = self.explainability.explain_service_selection(
                    self.service_selector.model, features, feature_names,
                    ranked_train["service_score"]
                )
                explanations[train_id] = explanation
        
        return {
            "date": date.isoformat(),
            "demand_forecasts": demand_forecasts,
            "risk_predictions": {k: v for k, v in list(risk_predictions.items())[:5]},  # Sample
            "ranked_trains": ranked_trains,
            "stabling_allocations": stabling_allocations,
            "transfer_decisions": transfer_decisions,
            "shunting_schedules": shunting_schedules,
            "explanations": explanations,
        }
    
    def _create_state(self, depot_configs: List[DepotConfig],
                     fleet_features_list: List[FleetFeatures],
                     risk_predictions: Dict[str, Dict[str, Any]],
                     predicted_demand: Dict[str, int]) -> MultiDepotState:
        """Create multi-depot state representation"""
        # Initialize depot states
        depot_bay_occupancy = {}
        depot_train_health_scores = {}
        depot_turnout_times = {}
        train_locations = {}
        train_decisions = {}
        
        for depot_config in depot_configs:
            depot_bay_occupancy[depot_config.depot_id] = {
                bay_id: None for bay_id in range(1, depot_config.total_bays + 1)
            }
            depot_train_health_scores[depot_config.depot_id] = {}
            depot_turnout_times[depot_config.depot_id] = {}
        
        # Populate from fleet features
        for fleet_features in fleet_features_list:
            train_id = fleet_features.train_id
            current_location = fleet_features.current_location
            depot_id = current_location.get("depot", depot_configs[0].depot_id)
            
            train_locations[train_id] = depot_id
            train_decisions[train_id] = "STANDBY"  # Default
            
            # Health score
            risk_pred = risk_predictions.get(train_id, {})
            health_score = risk_pred.get("health_score", 0.85)
            depot_train_health_scores[depot_id][train_id] = health_score
        
        # Distance matrix (simplified - would use actual coordinates)
        num_depots = len(depot_configs)
        distance_matrix = np.ones((num_depots, num_depots)) * 20.0  # Default 20 km
        np.fill_diagonal(distance_matrix, 0.0)
        
        return MultiDepotState(
            depot_bay_occupancy=depot_bay_occupancy,
            depot_train_health_scores=depot_train_health_scores,
            depot_turnout_times=depot_turnout_times,
            distance_matrix=distance_matrix,
            predicted_demand=predicted_demand,
            train_locations=train_locations,
            train_decisions=train_decisions,
        )
    
    def _convert_to_shunting_operations(self, allocations: List[Dict[str, Any]],
                                       depot_config: DepotConfig) -> List[Dict[str, Any]]:
        """Convert allocations to shunting operations format"""
        operations = []
        
        for alloc in allocations:
            allocation = alloc.get("allocation", {})
            if allocation.get("depot_id") != depot_config.depot_id:
                continue
            
            bay_id = allocation.get("bay_id")
            if bay_id:
                operations.append({
                    "trainset_id": alloc["train_id"],
                    "to_bay_id": bay_id,
                    "estimated_time_min": 10,  # Placeholder
                    "distance_m": 50,  # Placeholder
                    "complexity": "LOW",
                })
        
        return operations
    
    def _aggregate_metrics(self, daily_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all days"""
        total_service_trains = sum(
            len([t for t in day.get("ranked_trains", []) if t.get("selected")])
            for day in daily_results
        )
        
        total_transfers = sum(len(day.get("transfer_decisions", [])) for day in daily_results)
        
        total_shunting_time = 0
        for day in daily_results:
            for depot_schedule in day.get("shunting_schedules", {}).values():
                total_shunting_time += sum(
                    op.get("estimated_time_min", 10) for op in depot_schedule
                )
        
        return {
            "total_service_trains": total_service_trains,
            "total_transfers": total_transfers,
            "total_shunting_time_min": total_shunting_time,
            "avg_shunting_time_per_day": total_shunting_time / len(daily_results) if daily_results else 0,
        }
    
    async def _save_simulation_run(self, results: Dict[str, Any]):
        """Save simulation run to database"""
        try:
            collection = await cloud_db_manager.get_collection("simulation_runs")
            await collection.insert_one({
                "simulation_id": results["simulation_id"],
                "timestamp": results["timestamp"],
                "config": {
                    "depot_configs": results["depot_configs"],
                    "sim_days": results["sim_days"],
                    "seed": results["seed"],
                },
                "results": results,
                "metrics": results.get("aggregated_metrics", {}),
            })
            
            logger.info(f"Saved simulation run {results['simulation_id']}")
            
        except Exception as e:
            logger.error(f"Error saving simulation run: {e}")


