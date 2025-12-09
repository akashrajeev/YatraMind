# Multi-Depot AI-Driven Simulation Engine

## Overview

The YatraMind/KMRL metro fleet system now includes a fully AI-driven Multi-Depot Simulation Engine that replaces rule-heavy decisions with learned policies and predictive models. The system uses machine learning and reinforcement learning to make intelligent decisions about depot selection, inter-depot transfers, stabling, shunting, and rollout.

## Architecture

### Core Components

#### A) Data/Config Layer (`app/ml/multi_depot/config.py`)
- **DepotConfig**: Exposes depot features (bay counts, shunting_graph, turnout_map, maintenance_capacity)
- **FleetFeatures**: Exposes fleet features (train_id, mileage, sensor timeseries, job-cards, branding, historical failures)
- **SimulationRun**: Entity for simulation runs with seed, config, date_range
- **MultiDepotState**: State representation for RL agents

#### B) Predictive Models

1. **FailureRiskModel** (`failure_risk_model.py`)
   - LSTM/Transformer time-series model
   - Inputs: Last N days of IoT telemetry, mileage, job-cards, FC events
   - Outputs: P(failure in 24h), P(failure in 72h), component-level risk
   - Persists predictions per train per run

2. **DemandForecaster** (`demand_forecaster.py`)
   - Gradient Boosting (LightGBM) or sequence model
   - Predicts service demand bands and required service_trains per band
   - Used to compute required_service_trains per depot/network

#### C) Service Selection Model (`service_selection_model.py`)
- Supervised ranker (neural network)
- Features: failure_risk, branding_priority, recent uptime, cleaning status, last_turnout_time, dead_km cost
- Uses softmax sampling + top-K selection
- Allows temperature for stochasticity in training

#### D) RL Multi-Depot Stabling Agent (`rl_stabling_agent.py`)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **State**: Vectorized view of all depots (bay occupancy, train health scores, turnout_time_map, distance matrix, predicted demand)
- **Action**: Assign train → {depot_id, location_type: bay/terminal/yard, bay_slot}
- **Reward**: Weighted sum:
  - +service_coverage (maximize)
  - -failure_risk_penalty (minimize)
  - -total_dead_km (minimize)
  - -shunting_time (minimize)
  - Heavy penalty for infeasible assignments (bay overflows, heavy maintenance at terminal)
- **Training**: Curriculum learning (start 1 depot/25 trains, scale to N depots/100 trains)

#### E) RL Shunting Sequencer (`rl_shunting_sequencer.py`)
- Per-depot agent
- **State**: Bay_diff graph + current shunting window remaining + available crew
- **Action**: Pick next move or batch-move
- **Reward**: Negative of cumulative shunting time + window-violation penalty

#### F) Inter-Depot Transfer Decider (`transfer_decider.py`)
- Learned binary decision per candidate transfer
- Uses predicted demand + failure risk
- Cost model: transfer_dead_km + downtime vs expected improvement in service reliability

#### G) Feedback Loop (`feedback_loop.py`)
- Logs outcomes after every simulated/production day:
  - Actual failures
  - Realized shunting_time
  - Rollout delays
  - Branding exposure delivered
- Retrains models:
  - FailureRiskModel → incremental nightly/weekly
  - Service ranker → weekly
  - RL agents → continuous training in background

#### H) Explainability & Safety (`explainability.py`)
- SHAP/local explainers for all ML/decision outputs
- Surfaces top-3 contributing features
- **Safety Guard**: Hard constraints enforced at runtime
  - Rejects AI actions that violate Tier-1 safety constraints
  - Logs counterfactuals for training

#### I) Simulation Engine (`simulation_engine.py`)
- Main integration module
- Coordinates all AI modules
- Runs end-to-end simulation

## API Endpoints

### `/api/v1/multi-depot/`

- **POST `/simulate`** - Run multi-depot simulation
  - Parameters: `depot_configs[]`, `train_count`, `sim_days`, `seed`
  - Returns: Full multi-depot plan with AI decisions, allocations, transfers, schedules

- **GET `/explain`** - Get AI explanation for decision
  - Parameters: `train_id`, `decision_type`
  - Returns: SHAP-style attribution with top-3 contributing features

- **GET `/policy-status`** - Get model versions, training stats, last retrain time

- **POST `/feedback/log`** - Log production outcomes

- **POST `/feedback/process`** - Process feedback and retrain models

## Usage Example

```python
from app.ml.multi_depot.simulation_engine import MultiDepotSimulationEngine
from app.ml.multi_depot.config import DepotConfig, LocationType

# Create depot configs
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
    DepotConfig(
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
    ),
]

# Initialize engine
engine = MultiDepotSimulationEngine()
await engine.initialize(depot_configs)

# Run simulation
results = await engine.simulate(
    depot_configs=depot_configs,
    fleet_features_list=fleet_features,
    sim_days=1,
    seed=42,
)

# Results include:
# - Demand forecasts per depot
# - Risk predictions for all trains
# - Ranked service selections with probabilities
# - Stabling allocations with explanations
# - Inter-depot transfer recommendations
# - Optimized shunting schedules per depot
# - AI explanations for top decisions
```

## Training

### Initial Training

Models can be trained using historical data:

```python
from app.ml.multi_depot.training_orchestrator import TrainingOrchestrator

orchestrator = TrainingOrchestrator()

# Train all models
results = await orchestrator.train_all_models({
    "train_failure_risk": True,
    "train_demand_forecast": True,
    "train_service_selection": True,
    "failure_risk_epochs": 50,
    "service_selection_epochs": 50,
})

# Train RL agents with curriculum
rl_results = await orchestrator.train_rl_agents(
    depot_configs=depot_configs,
    num_episodes=1000,
    curriculum=True,
)
```

### Continuous Learning

The feedback loop automatically:
1. Logs daily production outcomes
2. Processes feedback weekly
3. Retrains models incrementally
4. Updates RL policies continuously

## Safety & Explainability

### Safety Guards

All AI decisions are validated against hard constraints:
- Fitness certificates must be valid
- No critical job cards
- Mileage within maintenance limits
- Maintenance status checks

If an AI action violates safety, it's rejected and logged for training.

### Explainability

Every AI decision includes:
- **Top 3 contributing features** (SHAP values)
- **Human-readable explanation**
- **Confidence scores**

Example:
```json
{
  "train_id": "T-001",
  "decision": "INDUCT",
  "explanation": "Selected for service (score: 0.85). Top factors: Low Failure Risk, High Branding Priority, Recent Maintenance OK",
  "top_factors": [
    {"feature": "failure_risk_24h", "value": 0.03, "impact": "increases_score"},
    {"feature": "branding_priority", "value": 0.8, "impact": "increases_score"},
    {"feature": "sensor_health", "value": 0.92, "impact": "increases_score"}
  ]
}
```

## Performance

- **Inference Latency**: < 500ms per train for UI interactivity
- **RL Agent Inference**: < 3s for full run (batched)
- **Fallback**: Rule-based solver if AI services unavailable

## Demo

Run the demo script to see baseline vs multi-depot AI comparison:

```bash
python backend/scripts/demo_multi_depot.py
```

This will:
1. Run baseline: 1 depot, 25 trains
2. Run multi-depot AI: 2 depots, 40 trains
3. Compare metrics (service coverage, shunting time, failure events)
4. Show AI explanations

## Model Cards

### Failure Risk Model
- **Type**: LSTM/Transformer time-series
- **Input**: 30 days of sensor telemetry, mileage, job cards
- **Output**: Risk probabilities (24h, 72h), component-level risks
- **Training**: Supervised learning with historical failure data

### Service Selection Model
- **Type**: Neural network ranker
- **Input**: Risk predictions, branding, reliability, turnout time
- **Output**: Service selection probability
- **Training**: Supervised learning with actual selection outcomes

### RL Stabling Agent
- **Algorithm**: PPO (Proximal Policy Optimization)
- **State Space**: Multi-depot state vector
- **Action Space**: Depot + location + bay assignment
- **Reward Function**: Service coverage - risks - costs
- **Training**: Curriculum learning, thousands of episodes

## Reward Function Details

### Stabling Agent Reward
```
reward = +10.0 * service_coverage
         -20.0 * failure_risk
         -0.5 * dead_km
         -0.2 * shunting_time_min
         -50.0 * infeasible_penalty
         -30.0 * bay_overflow_penalty
         -40.0 * maintenance_violation_penalty
```

### Shunting Sequencer Reward
```
reward = -(time_penalty + overrun_penalty + conflict_penalty)
where:
  time_penalty = total_time / available_window
  overrun_penalty = (total_time - window) / window * 10.0
  conflict_penalty = bay_conflicts * 5.0
```

## Safety Guard Details

### Tier-1 Safety Constraints (Hard)
1. **Fitness Certificates**: All 3 required (Rolling-Stock, Signalling, Telecom) must be VALID
2. **Job Cards**: No critical job cards open
3. **Mileage**: Within maintenance limits
4. **Maintenance Status**: Not currently in maintenance

If AI proposes action violating these, it's **automatically rejected** and logged.

## Testing

Run integration tests:

```bash
pytest backend/tests/test_multi_depot_simulation.py
```

Tests verify:
- No Tier-1 safety violations
- Effective service shortfall <= baseline
- RL policy reduces average shunting_time vs baseline in >70% of scenarios

## Future Enhancements

1. **Online Learning**: Real-time model updates
2. **Multi-Agent RL**: Coordinated multi-agent optimization
3. **Transfer Learning**: Pre-trained models for faster deployment
4. **Federated Learning**: Distributed training across depots
5. **Explainable RL**: Better RL decision explanations


