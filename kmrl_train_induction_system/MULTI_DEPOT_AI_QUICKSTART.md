# Multi-Depot AI System - Quick Start Guide

## Quick Start

### 1. Run Demo

```bash
cd backend
python scripts/demo_multi_depot.py
```

This demonstrates:
- Baseline: 1 depot, 25 trains
- Multi-Depot AI: 2 depots, 40 trains
- Metrics comparison
- AI explanations

### 2. Use API

```bash
# Run simulation
curl -X POST http://localhost:8000/api/v1/multi-depot/simulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "depot_configs": [
      {
        "depot_id": "MUTTOM",
        "depot_name": "Muttom Depot",
        "location_type": "FULL_DEPOT",
        "service_bay_capacity": 6,
        "maintenance_bay_capacity": 4,
        "standby_bay_capacity": 2,
        "total_bays": 12,
        "supports_heavy_maintenance": true,
        "supports_cleaning": true,
        "can_start_service": true
      },
      {
        "depot_id": "KAKKANAD",
        "depot_name": "Kakkanad Depot",
        "location_type": "FULL_DEPOT",
        "service_bay_capacity": 4,
        "maintenance_bay_capacity": 3,
        "standby_bay_capacity": 2,
        "total_bays": 9,
        "supports_heavy_maintenance": true,
        "supports_cleaning": true,
        "can_start_service": true
      }
    ],
    "train_count": 40,
    "sim_days": 1,
    "seed": 42
  }'
```

### 3. Get AI Explanations

```bash
curl "http://localhost:8000/api/v1/multi-depot/explain?train_id=T-001&decision_type=service_selection" \
  -H "X-API-Key: your-api-key"
```

### 4. Check Model Status

```bash
curl "http://localhost:8000/api/v1/multi-depot/policy-status" \
  -H "X-API-Key: your-api-key"
```

## Key Features

✅ **AI-Driven Decisions**: All decisions made by learned models, not rules
✅ **Multi-Depot Support**: Handles N depots with intelligent distribution
✅ **Continuous Learning**: Models improve with every day of operation
✅ **Full Explainability**: SHAP explanations for every decision
✅ **Safety First**: Hard constraints enforced, AI proposals validated
✅ **Scalable**: Handles 25-100 trains across multiple depots

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│           Multi-Depot Simulation Engine                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Failure    │  │   Demand     │  │   Service    │   │
│  │ Risk Model   │  │  Forecaster  │  │  Selector   │   │
│  │ (LSTM/Trans) │  │ (LightGBM)   │  │   (NN)       │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ RL Stabling  │  │ RL Shunting  │  │   Transfer   │   │
│  │   Agent      │  │  Sequencer   │  │   Decider    │   │
│  │   (PPO)      │  │   (PPO)      │  │    (NN)      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │     Feedback Loop & Online Learning              │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │     Explainability & Safety Guards               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Train Initial Models**: Use historical data to train initial models
2. **Start Logging**: Begin logging production outcomes daily
3. **Enable Feedback Loop**: Process feedback weekly to retrain
4. **Monitor Performance**: Track model metrics and improvements
5. **Scale Up**: Gradually increase fleet size and depots

## Performance Targets

- ✅ Inference: < 500ms per train
- ✅ Full Run: < 3s for RL agent
- ✅ Shunting Time: >70% improvement vs baseline
- ✅ Safety: 0 Tier-1 violations
- ✅ Explainability: Top-3 factors for every decision























