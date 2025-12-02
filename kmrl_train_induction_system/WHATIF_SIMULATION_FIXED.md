# What-If Simulation System - Complete Fix Summary

## Overview
The What-If Simulation system has been completely rebuilt to work reliably, deterministically, and with stable JSON structure. All fixes ensure that `results` is always an array to prevent frontend crashes.

---

## ✅ Backend Implementation

### 1. **Snapshot System** (`backend/app/utils/snapshot.py`)
- **Function**: `capture_snapshot()` - Captures complete system state
- **Features**:
  - Deep copies all data (no DB mutation)
  - Captures: trainsets, depot_layouts, cleaning_slots, certificates, jobcards, bay_locations, config
  - Deterministic and reproducible

### 2. **What-If Simulator** (`backend/app/services/whatif_simulator.py`)
- **Function**: `run_whatif(scenario, snapshot=None)`
- **Features**:
  - Captures baseline snapshot if not provided
  - Applies scenario overrides (path-based nested setters)
  - Runs baseline and scenario optimizations in-memory (no DB writes)
  - Computes KPIs for both baseline and scenario
  - Calculates deltas (scenario - baseline)
  - Generates explain_log with human-readable changes
  - **ALWAYS returns `results` as array** `[baseline_result, scenario_result]`
  - Saves results to `backend/simulation_runs/<uuid>.json`
  - Supports `random_seed` for deterministic execution

### 3. **Scenario Override System**
- **Supported Overrides**:
  - `required_service_hours`: Override service hours requirement
  - `override_train_attributes`: Path-based nested setter (e.g., `{"T-001": {"fitness.telecom.status": "EXPIRED"}}`)
  - `depot_layout_override`: Override depot layouts
  - `cleaning_capacity_override`: Override cleaning capacity
  - `force_decisions`: Force specific trainset decisions (e.g., `{"T-001": "INDUCT"}`)
  - `inject_delay_events`: Inject delay events
  - `random_seed`: Seed for deterministic randomness

### 4. **API Routes** (`backend/app/api/simulation.py`)
- **POST `/api/simulation/run`**: Run What-If simulation immediately
- **GET `/api/simulation/result/{id}`**: Load saved simulation result
- **GET `/api/simulation/snapshot`**: Get current system snapshot
- **All routes ensure `results` is always an array** via `_ensure_results_is_array()`

### 5. **Fixed Existing Endpoint**
- **GET `/api/optimization/simulate`**: Updated to use new system (backwards compatible)

---

## ✅ Frontend Implementation

### 1. **Safe Results Utility** (`frontend/src/utils/simulation.ts`)
- **Functions**:
  - `ensureResultsArray(results)`: Converts any results to array
  - `getBaselineResult(simulationData)`: Gets baseline from results array
  - `getScenarioResult(simulationData)`: Gets scenario from results array
  - `filterResults()`, `mapResults()`, `getResultsLength()`: Safe array operations

### 2. **Updated UI** (`frontend/src/pages/Optimization.tsx`)
- Uses `ensureResultsArray()` to safely handle results
- Displays baseline vs scenario comparison
- Shows deltas with color coding (green for improvements, red for degradations)
- Displays explain_log with human-readable explanations
- Backwards compatible with legacy format

### 3. **API Service** (`frontend/src/services/api.ts`)
- Added:
  - `runSimulation(scenario)`: POST to `/api/simulation/run`
  - `getSimulationResult(id)`: GET from `/api/simulation/result/{id}`
  - `getSnapshot()`: GET from `/api/simulation/snapshot`

---

## ✅ Key Features

### Determinism
- **No randomness** unless `random_seed` is explicitly provided
- Same scenario → same output (excluding timestamps and IDs)
- ML predictions are seeded for reproducibility

### Results Array Guarantee
- **Backend**: Always returns `results` as array `[baseline_result, scenario_result]`
- **Frontend**: Uses `ensureResultsArray()` to handle any format safely
- **No crashes**: Prevents "filter is not a function" errors

### Numeric Fields
- All numeric fields are explicitly converted to `int` or `float`
- No string parsing (e.g., "8 minutes" → numeric `8`)
- `total_shunting_time`, `estimated_time` are always numeric

### In-Memory Execution
- **No DB mutations**: All simulations run in-memory
- Uses deep copies of snapshots
- Safe for production use

---

## ✅ Test Coverage

All tests in `backend/tests/whatif/`:

1. **test_snapshot_capture.py**: Snapshot contains required keys
2. **test_run_whatif_basic.py**: Simple scenario, ensure results array, no crashes
3. **test_results_always_array.py**: Mock incorrectly-shaped results and ensure coercion to array
4. **test_determinism.py**: Same scenario twice returns identical output
5. **test_override_path_setter.py**: Path setter modifies snapshot correctly
6. **test_scenario_vs_baseline_kpis.py**: Deltas computed correctly
7. **test_load_saved_simulation.py**: Saved JSON loads correctly

---

## 📊 Response Structure

```json
{
  "simulation_id": "uuid",
  "timestamp": "2024-01-01T00:00:00",
  "baseline": {
    "num_inducted_trains": 10,
    "total_shunting_time": 120,
    "efficiency_improvement": 75.0,
    ...
  },
  "scenario": {
    "num_inducted_trains": 12,
    "total_shunting_time": 80,
    "efficiency_improvement": 80.0,
    ...
  },
  "deltas": {
    "num_inducted_trains": 2,
    "total_shunting_time": -40,
    "efficiency_improvement": 5.0,
    ...
  },
  "explain_log": [
    "Inducted trains changed by +2 (baseline: 10, scenario: 12)",
    "Total shunting time changed by -40 minutes (baseline: 120, scenario: 80)",
    ...
  ],
  "results": [
    {
      "type": "baseline",
      "kpis": {...},
      "decisions": [...],
      "stabling_geometry": {...}
    },
    {
      "type": "scenario",
      "kpis": {...},
      "decisions": [...],
      "stabling_geometry": {...}
    }
  ]
}
```

**CRITICAL**: `results` is **ALWAYS** an array, never an object.

---

## 🚀 Usage Examples

### Basic Simulation
```python
scenario = {
    "required_service_hours": 16
}
result = await run_whatif(scenario)
```

### With Overrides
```python
scenario = {
    "required_service_hours": 16,
    "override_train_attributes": {
        "T-001": {
            "fitness_certificates.telecom.status": "EXPIRED",
            "current_mileage": 50000
        }
    },
    "force_decisions": {
        "T-002": "INDUCT"
    }
}
result = await run_whatif(scenario)
```

### Deterministic with Seed
```python
scenario = {
    "required_service_hours": 16,
    "random_seed": 42
}
result = await run_whatif(scenario)
```

---

## ✅ Verification Checklist

- [x] Snapshot system captures all required data
- [x] What-If simulator runs baseline and scenario
- [x] Results always returned as array
- [x] Deltas computed correctly
- [x] Explain log generated
- [x] Results saved to file
- [x] API routes work correctly
- [x] Frontend safely handles results
- [x] No randomness (unless seed provided)
- [x] All numeric fields are numeric
- [x] No DB mutations
- [x] All tests pass

---

## 🎯 Summary

The What-If Simulation system is now:
- ✅ **Reliable**: No crashes, always returns valid structure
- ✅ **Deterministic**: Same input → same output
- ✅ **Safe**: No DB mutations, in-memory execution
- ✅ **Stable**: Results always array, numeric fields are numeric
- ✅ **Complete**: Baseline vs scenario comparison with deltas and explanations
- ✅ **Tested**: Comprehensive test coverage

The system is production-ready and will not cause frontend crashes.







