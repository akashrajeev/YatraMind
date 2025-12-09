#!/usr/bin/env python3
"""
Demo Script: Multi-Depot AI Simulation
Run 1-depot-25-train baseline vs 2-depot-40-train simulation and show metrics
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.multi_depot.simulation_engine import MultiDepotSimulationEngine
from app.ml.multi_depot.config import DepotConfig, FleetFeatures, LocationType


async def create_demo_depot_configs() -> list:
    """Create demo depot configurations"""
    muttom = DepotConfig(
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
        coordinates=(10.1081, 76.3527),  # Approximate
    )
    
    kakkanad = DepotConfig(
        depot_id="KAKKANAD",
        depot_name="Kakkanad Depot",
        location_type=LocationType.FULL_DEPOT,
        service_bay_capacity=4,
        maintenance_bay_capacity=3,
        standby_bay_capacity=2,
        total_bays=9,
        supports_heavy_maintenance=True,
        supports_cleaning=True,
        can_start_service=True,
        coordinates=(10.0169, 76.3415),  # Approximate
    )
    
    return [muttom, kakkanad]


async def create_demo_fleet(count: int = 25) -> list:
    """Create demo fleet features"""
    fleet = []
    for i in range(1, count + 1):
        train_id = f"T-{i:03d}"
        fleet.append(FleetFeatures(
            train_id=train_id,
            mileage=10000.0 + i * 1000,
            job_cards={"critical_cards": 0, "open_cards": i % 5},
            branding_flag=(i % 3 == 0),
            branding_priority=0.8 if i % 3 == 0 else 0.0,
            sensor_health_score=0.85 + (i % 10) * 0.01,
            current_mileage=10000.0 + i * 1000,
            max_mileage_before_maintenance=50000.0,
            current_location={"depot": "MUTTOM", "bay": f"BAY_{i % 12 + 1}"},
        ))
    return fleet


async def run_baseline_simulation():
    """Run baseline: 1 depot, 25 trains"""
    print("=" * 60)
    print("BASELINE SIMULATION: 1 Depot (Muttom), 25 Trains")
    print("=" * 60)
    
    engine = MultiDepotSimulationEngine()
    depot_configs = [await create_demo_depot_configs()[0]]  # Just Muttom
    fleet = await create_demo_fleet(25)
    
    results = await engine.simulate(
        depot_configs=depot_configs,
        fleet_features_list=fleet,
        sim_days=1,
        seed=42,
    )
    
    metrics = results.get("aggregated_metrics", {})
    print(f"\nBaseline Metrics:")
    print(f"  Total Service Trains: {metrics.get('total_service_trains', 0)}")
    print(f"  Total Shunting Time: {metrics.get('total_shunting_time_min', 0)} min")
    print(f"  Avg Shunting Time/Day: {metrics.get('avg_shunting_time_per_day', 0):.1f} min")
    
    return results


async def run_multi_depot_simulation():
    """Run multi-depot: 2 depots, 40 trains"""
    print("\n" + "=" * 60)
    print("MULTI-DEPOT AI SIMULATION: 2 Depots (Muttom + Kakkanad), 40 Trains")
    print("=" * 60)
    
    engine = MultiDepotSimulationEngine()
    depot_configs = await create_demo_depot_configs()  # Both depots
    fleet = await create_demo_fleet(40)
    
    results = await engine.simulate(
        depot_configs=depot_configs,
        fleet_features_list=fleet,
        sim_days=1,
        seed=42,
    )
    
    metrics = results.get("aggregated_metrics", {})
    print(f"\nMulti-Depot AI Metrics:")
    print(f"  Total Service Trains: {metrics.get('total_service_trains', 0)}")
    print(f"  Total Transfers: {metrics.get('total_transfers', 0)}")
    print(f"  Total Shunting Time: {metrics.get('total_shunting_time_min', 0)} min")
    print(f"  Avg Shunting Time/Day: {metrics.get('avg_shunting_time_per_day', 0):.1f} min")
    
    # Show AI explanations (sample)
    if results.get("daily_results"):
        day_result = results["daily_results"][0]
        explanations = day_result.get("explanations", {})
        if explanations:
            print(f"\nAI Explanations (sample):")
            for train_id, explanation in list(explanations.items())[:3]:
                print(f"  {train_id}: {explanation.get('explanation', 'N/A')}")
    
    return results


async def main():
    """Run demo comparison"""
    print("\n" + "=" * 60)
    print("MULTI-DEPOT AI SIMULATION DEMO")
    print("=" * 60)
    
    # Run baseline
    baseline_results = await run_baseline_simulation()
    
    # Run multi-depot AI
    ai_results = await run_multi_depot_simulation()
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Multi-Depot AI")
    print("=" * 60)
    
    baseline_metrics = baseline_results.get("aggregated_metrics", {})
    ai_metrics = ai_results.get("aggregated_metrics", {})
    
    baseline_shunting = baseline_metrics.get("total_shunting_time_min", 0)
    ai_shunting = ai_metrics.get("total_shunting_time_min", 0)
    
    improvement = ((baseline_shunting - ai_shunting) / baseline_shunting * 100) if baseline_shunting > 0 else 0
    
    print(f"\nShunting Time Improvement: {improvement:.1f}%")
    print(f"  Baseline: {baseline_shunting} min")
    print(f"  AI: {ai_shunting} min")
    print(f"  Reduction: {baseline_shunting - ai_shunting} min")
    
    print(f"\nService Coverage:")
    print(f"  Baseline: {baseline_metrics.get('total_service_trains', 0)} trains")
    print(f"  AI: {ai_metrics.get('total_service_trains', 0)} trains")
    
    print(f"\nInter-Depot Transfers: {ai_metrics.get('total_transfers', 0)}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


