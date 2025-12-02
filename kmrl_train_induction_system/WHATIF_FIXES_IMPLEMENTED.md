# What-If Simulation Fixes - Implementation Summary

## ✅ Fixes Implemented

### 1. **Frontend API Endpoint Fix** ✅
**File:** `frontend/src/pages/Optimization.tsx`

**Changed:**
- ❌ **Before:** `mutationFn: optimizationApi.simulate` (GET `/optimization/simulate`)
- ✅ **After:** `mutationFn: (params) => optimizationApi.runSimulation(transformSimulationParams(params))` (POST `/simulation/run`)

**Impact:** Frontend now calls the correct, modern endpoint.

---

### 2. **Parameter Transformation** ✅
**File:** `frontend/src/pages/Optimization.tsx`

**Added `transformSimulationParams()` function:**
- Converts `required_service_count` → `required_service_hours`
- Parses `force_induct` (comma-separated) → `force_decisions` dict
- Parses `exclude_trainsets` (comma-separated) → `override_train_attributes` with expired fitness certs
- Converts weights object to match backend format

**Example Transformation:**
```typescript
// Frontend params:
{
  exclude_trainsets: "RK-001, RK-002",
  force_induct: "RK-003",
  required_service_count: 14,
  w_readiness: 0.35,
  w_reliability: 0.30,
  w_branding: 0.20,
  w_shunt: 0.10,
  w_mileage_balance: 0.05
}

// Transformed to backend scenario:
{
  required_service_hours: 14,
  force_decisions: { "RK-003": "INDUCT" },
  override_train_attributes: {
    "RK-001": { "fitness_certificates.rolling_stock.status": "EXPIRED" },
    "RK-002": { "fitness_certificates.rolling_stock.status": "EXPIRED" }
  },
  weights: {
    readiness: 0.35,
    reliability: 0.30,
    branding: 0.20,
    shunt: 0.10,
    mileage_balance: 0.05
  }
}
```

---

### 3. **Backend Weights Support** ✅
**File:** `backend/app/api/simulation.py`

**Added to `WhatIfScenario` model:**
```python
weights: Optional[Dict[str, float]] = Field(
    None,
    description="Optimization weights (e.g., {'readiness': 0.35, 'reliability': 0.30, ...})"
)
```

**Impact:** Backend now accepts and validates weights from frontend.

---

### 4. **Weights Usage in Simulation** ✅
**File:** `backend/app/services/whatif_simulator.py`

**Updated `run_whatif()` function:**
- Extracts weights from scenario if provided
- Creates `OptimizationWeights` object from scenario weights
- Applies weights to scenario optimization (not baseline)
- Baseline always uses default weights for fair comparison

**Code:**
```python
# Create weights from scenario if provided
scenario_weights = None
if "weights" in scenario and scenario["weights"]:
    try:
        scenario_weights = OptimizationWeights(**scenario["weights"])
        logger.info(f"Using custom weights from scenario: {scenario['weights']}")
    except Exception as e:
        logger.warning(f"Invalid weights in scenario, using defaults: {e}")
        scenario_weights = OptimizationWeights()
else:
    scenario_weights = OptimizationWeights()

# Scenario uses custom weights
request = OptimizationRequest(
    target_date=datetime.utcnow().date(),
    required_service_hours=int(required_hours),
    weights=scenario_weights  # Custom weights applied
)

# Baseline uses default weights
baseline_request = OptimizationRequest(
    target_date=datetime.utcnow().date(),
    required_service_hours=int(snapshot["config"].get("required_service_hours", 14)),
    weights=OptimizationWeights()  # Default weights
)
```

---

## 📋 Files Changed

### Frontend
1. **`frontend/src/pages/Optimization.tsx`**
   - Added `transformSimulationParams()` function
   - Updated `runSimulationMutation` to use `runSimulation` endpoint
   - Added error handling in mutation

### Backend
2. **`backend/app/api/simulation.py`**
   - Added `weights` field to `WhatIfScenario` model

3. **`backend/app/services/whatif_simulator.py`**
   - Added weights extraction and validation
   - Applied weights to scenario optimization
   - Baseline uses default weights

---

## 🧪 Testing Checklist

### Manual Testing Steps

1. **Test Basic Simulation**
   - [ ] Open What-If Simulation tab
   - [ ] Set required service count (e.g., 14)
   - [ ] Click "Run What-If Simulation"
   - [ ] Verify results appear without errors

2. **Test Force Induct**
   - [ ] Enter trainset IDs in "Force Induct" field (e.g., "RK-001, RK-002")
   - [ ] Run simulation
   - [ ] Verify those trainsets are inducted in scenario

3. **Test Exclude Trainsets**
   - [ ] Enter trainset IDs in "Exclude Trainsets" field (e.g., "RK-003, RK-004")
   - [ ] Run simulation
   - [ ] Verify those trainsets are excluded from scenario

4. **Test Weights**
   - [ ] Modify optimization weights (e.g., set readiness to 0.5)
   - [ ] Run simulation
   - [ ] Verify scenario results differ from baseline
   - [ ] Verify baseline uses default weights

5. **Test Combined Parameters**
   - [ ] Set all parameters (exclude, force, count, weights)
   - [ ] Run simulation
   - [ ] Verify all parameters are applied correctly

6. **Test Error Handling**
   - [ ] Run simulation with invalid data
   - [ ] Verify error message appears
   - [ ] Verify UI doesn't crash

---

## 🔍 Verification

### API Endpoint Verification
```bash
# Should now call:
POST /api/simulation/run
Content-Type: application/json

{
  "required_service_hours": 14,
  "force_decisions": {"RK-003": "INDUCT"},
  "override_train_attributes": {
    "RK-001": {"fitness_certificates.rolling_stock.status": "EXPIRED"}
  },
  "weights": {
    "readiness": 0.35,
    "reliability": 0.30,
    "branding": 0.20,
    "shunt": 0.10,
    "mileage_balance": 0.05
  }
}
```

### Expected Response
```json
{
  "simulation_id": "uuid-here",
  "timestamp": "2024-...",
  "baseline": { "num_inducted_trains": 14, ... },
  "scenario": { "num_inducted_trains": 15, ... },
  "deltas": { "num_inducted_trains": 1, ... },
  "explain_log": ["..."],
  "results": [
    { "type": "baseline", "kpis": {...}, "decisions": [...] },
    { "type": "scenario", "kpis": {...}, "decisions": [...] }
  ]
}
```

---

## 🎯 Key Improvements

1. **Correct API Usage**: Frontend now uses the modern `/simulation/run` endpoint
2. **Parameter Compatibility**: Frontend parameters are properly transformed to backend format
3. **Weights Support**: User-set weights are now applied to scenario optimization
4. **Error Handling**: Added error handling in frontend mutation
5. **Backwards Compatible**: Old `/optimization/simulate` endpoint still works for legacy code

---

## 📝 Notes

- **Baseline vs Scenario**: Baseline always uses default weights for fair comparison
- **Weights Validation**: Invalid weights fall back to defaults with warning log
- **Parameter Parsing**: Comma-separated lists are trimmed and filtered for empty values
- **Exclusion Method**: Excluded trainsets are marked with expired fitness certificates (proper way would be to add an exclusion flag, but this works for now)

---

## 🚀 Next Steps (Optional)

1. **Add Toast Notifications**: Show success/error messages to user
2. **Add Loading States**: Better UX during simulation
3. **Add Parameter Validation**: Validate trainset IDs exist before simulation
4. **Improve Exclusion Method**: Add proper exclusion flag instead of expired certs
5. **Add Progress Updates**: Show progress for long-running simulations

---

## ✅ Status

All critical fixes have been implemented. The What-If Simulation should now work correctly with:
- ✅ Correct API endpoint
- ✅ Proper parameter transformation
- ✅ Weights support
- ✅ Error handling

**Ready for testing!**







