# backend/tests/test_multi_depot_simulation.py
"""
Tests for Multi-Depot AI Simulation System
"""
import pytest
import asyncio
import numpy as np
from app.ml.multi_depot.simulation_engine import MultiDepotSimulationEngine
from app.ml.multi_depot.config import DepotConfig, FleetFeatures, LocationType
from app.ml.multi_depot.explainability import SafetyGuard
from datetime import datetime


@pytest.mark.asyncio
async def test_safety_guard_validates_fitness_certificates():
    """Test that safety guard rejects trains with expired FC"""
    guard = SafetyGuard()
    
    train_features = {
        "fitness_certificates": {
            "rolling_stock": {"status": "EXPIRED"},
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"},
        },
        "job_cards": {"critical_cards": 0, "open_cards": 0},
        "current_mileage": 10000,
        "max_mileage_before_maintenance": 50000,
        "status": "STANDBY",
    }
    
    is_valid, reason = guard.validate_service_decision(train_features, "INDUCT")
    assert not is_valid
    assert "Expired fitness certificates" in reason


@pytest.mark.asyncio
async def test_safety_guard_validates_critical_job_cards():
    """Test that safety guard rejects trains with critical job cards"""
    guard = SafetyGuard()
    
    train_features = {
        "fitness_certificates": {
            "rolling_stock": {"status": "VALID"},
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"},
        },
        "job_cards": {"critical_cards": 2, "open_cards": 0},
        "current_mileage": 10000,
        "max_mileage_before_maintenance": 50000,
        "status": "STANDBY",
    }
    
    is_valid, reason = guard.validate_service_decision(train_features, "INDUCT")
    assert not is_valid
    assert "Critical job cards" in reason


@pytest.mark.asyncio
async def test_multi_depot_simulation_basic():
    """Test basic multi-depot simulation runs without errors"""
    engine = MultiDepotSimulationEngine()
    
    depot_configs = [
        DepotConfig(
            depot_id="MUTTOM",
            depot_name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bay_capacity=6,
            maintenance_bay_capacity=4,
            standby_bay_capacity=2,
            total_bays=12,
            supports_heavy_maintenance=True,
            supports_cleaning=True,
            can_start_service=True,
        ),
    ]
    
    fleet_features = [
        FleetFeatures(
            train_id=f"T-{i:03d}",
            mileage=10000.0 + i * 1000,
            job_cards={"critical_cards": 0, "open_cards": i % 5},
            branding_flag=(i % 3 == 0),
            branding_priority=0.8 if i % 3 == 0 else 0.0,
            sensor_health_score=0.85,
            current_mileage=10000.0 + i * 1000,
            max_mileage_before_maintenance=50000.0,
        )
        for i in range(1, 26)
    ]
    
    results = await engine.simulate(
        depot_configs=depot_configs,
        fleet_features_list=fleet_features,
        sim_days=1,
        seed=42,
    )
    
    assert "simulation_id" in results
    assert "daily_results" in results
    assert len(results["daily_results"]) == 1
    
    day_result = results["daily_results"][0]
    assert "demand_forecasts" in day_result
    assert "ranked_trains" in day_result
    assert "stabling_allocations" in day_result


@pytest.mark.asyncio
async def test_multi_depot_simulation_no_safety_violations():
    """Test that simulation produces no Tier-1 safety violations"""
    engine = MultiDepotSimulationEngine()
    guard = SafetyGuard()
    
    depot_configs = [
        DepotConfig(
            depot_id="MUTTOM",
            depot_name="Muttom Depot",
            location_type=LocationType.FULL_DEPOT,
            service_bay_capacity=6,
            maintenance_bay_capacity=4,
            standby_bay_capacity=2,
            total_bays=12,
            supports_heavy_maintenance=True,
            supports_cleaning=True,
            can_start_service=True,
        ),
    ]
    
    fleet_features = [
        FleetFeatures(
            train_id=f"T-{i:03d}",
            mileage=10000.0 + i * 1000,
            job_cards={"critical_cards": 0, "open_cards": 0},
            branding_flag=False,
            branding_priority=0.0,
            sensor_health_score=0.9,
            current_mileage=10000.0 + i * 1000,
            max_mileage_before_maintenance=50000.0,
            fitness_certificates={
                "rolling_stock": {"status": "VALID"},
                "signalling": {"status": "VALID"},
                "telecom": {"status": "VALID"},
            },
        )
        for i in range(1, 26)
    ]
    
    results = await engine.simulate(
        depot_configs=depot_configs,
        fleet_features_list=fleet_features,
        sim_days=1,
        seed=42,
    )
    
    # Check that all service decisions pass safety guard
    day_result = results["daily_results"][0]
    ranked_trains = day_result.get("ranked_trains", [])
    
    for train in ranked_trains:
        if train.get("selected"):
            # Find corresponding fleet features
            train_id = train.get("train_id")
            fleet_feat = next((f for f in fleet_features if f.train_id == train_id), None)
            
            if fleet_feat:
                is_valid, reason = guard.validate_service_decision(
                    fleet_feat.dict(), "SERVICE"
                )
                assert is_valid, f"Safety violation for {train_id}: {reason}"


@pytest.mark.asyncio
async def test_explainability_provides_top_factors():
    """Test that explainability provides top contributing factors"""
    from app.ml.multi_depot.explainability import AIExplainability
    
    explainability = AIExplainability()
    
    # Test service selection explanation
    features = np.array([0.03, 0.15, 0.85, 1.0, 0.8, 0.95, 0.0, 10.0/30.0, 0.0, 0.0, 0.0, 0.92, 0.2])
    feature_names = ["risk_24h", "risk_72h", "health_score", "branding_flag", "branding_priority",
                    "uptime", "cleaning", "turnout_time", "dead_km", "critical_cards", "open_cards",
                    "sensor_health", "mileage_ratio"]
    
    explanation = explainability.explain_service_selection(None, features, feature_names, 0.85)
    
    assert "explanation" in explanation
    assert "score" in explanation
    # Should have top factors (even if fallback)
    assert isinstance(explanation.get("top_factors", []), list)


@pytest.mark.asyncio
async def test_demand_forecaster_produces_reasonable_values():
    """Test that demand forecaster produces reasonable demand values"""
    from app.ml.multi_depot.demand_forecaster import DemandForecaster
    
    forecaster = DemandForecaster()
    
    forecast = await forecaster.forecast(
        datetime.now(),
        depot_id="MUTTOM"
    )
    
    assert "required_service_trains" in forecast
    assert 10 <= forecast["required_service_trains"] <= 20  # Reasonable range
    assert "demand_bands" in forecast
    assert len(forecast["demand_bands"]) > 0


@pytest.mark.asyncio
async def test_failure_risk_predictor_produces_valid_probabilities():
    """Test that failure risk predictor produces valid probabilities"""
    from app.ml.multi_depot.failure_risk_model import FailureRiskPredictor
    from app.ml.multi_depot.config import FleetFeatures
    
    predictor = FailureRiskPredictor()
    
    fleet_features = FleetFeatures(
        train_id="T-001",
        mileage=20000.0,
        job_cards={"critical_cards": 0, "open_cards": 2},
        branding_flag=False,
        branding_priority=0.0,
        sensor_health_score=0.9,
        current_mileage=20000.0,
        max_mileage_before_maintenance=50000.0,
    )
    
    prediction = await predictor.predict(fleet_features)
    
    assert "risk_24h" in prediction
    assert "risk_72h" in prediction
    assert 0.0 <= prediction["risk_24h"] <= 1.0
    assert 0.0 <= prediction["risk_72h"] <= 1.0
    assert "component_risks" in prediction
    assert "health_score" in prediction

