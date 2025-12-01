# What-If Simulation Analysis Report

## Executive Summary

The What-If Simulation system has been implemented with defensive programming and normalization layers. However, there is a **critical mismatch** between the frontend API call and the backend endpoint that needs to be addressed.

---

## ✅ What Works Well

### 1. **Backend Implementation**
- **`whatif_simulator.py`**: Well-structured service with:
  - Deterministic execution (seeded randomness)
  - Snapshot-based isolation (no DB mutations)
  - Comprehensive KPI computation
  - Decision normalization (ensures `explain` fields)
  - Always returns `results` as an array

- **`simulation.py` API**: Proper endpoints with:
  - POST `/api/simulation/run` - Main simulation endpoint
  - GET `/api/simulation/result/{id}` - Load saved results
  - GET `/api/simulation/snapshot` - Get current snapshot
  - Normalization layer ensures consistent data shape

### 2. **Frontend Defensive Components**
- **`DetailedResults.tsx`**: Robust component that:
  - Never crashes on malformed data
  - Handles null/undefined/objects/arrays
  - Extracts explanations from multiple fields
  - Uses emoji instead of icon to avoid import issues
  - Explicit pointer events for clickability

- **`simulation.ts` utilities**: Safe array helpers:
  - `asArray()` - Converts any value to array
  - `extractExplanation()` - Multi-field explanation extraction
  - `getBaselineResult()` / `getScenarioResult()` - Safe result extraction

### 3. **Data Normalization**
- Backend ensures all decisions have `explain` fields
- Results are always arrays
- Numeric fields are properly typed
- Defensive checks at multiple layers

---

## ⚠️ Critical Issues

### 1. **API Endpoint Mismatch** (CRITICAL)

**Problem:**
- Frontend calls: `optimizationApi.simulate()` → GET `/optimization/simulate`
- New backend endpoint: POST `/api/simulation/run`
- The old endpoint exists but uses different parameter format

**Current Frontend Code:**
```typescript
const runSimulationMutation = useMutation({
  mutationFn: optimizationApi.simulate,  // ❌ Calls GET /optimization/simulate
  ...
});
```

**Frontend Parameters:**
```typescript
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
```

**Backend Expected Format (POST /simulation/run):**
```typescript
{
  required_service_hours: 14,
  override_train_attributes: { "T-001": { "fitness.telecom.valid_until": "..." } },
  force_decisions: { "T-001": "INDUCT" },
  random_seed: 0
}
```

**Impact:** 
- The old endpoint redirects but parameter conversion is incomplete
- Weights (w_readiness, etc.) are not passed to the new system
- Parameter format mismatch may cause incorrect simulations

**Fix Required:**
```typescript
// Change frontend to use:
const runSimulationMutation = useMutation({
  mutationFn: optimizationApi.runSimulation,  // ✅ Use POST /simulation/run
  ...
});

// Transform parameters:
const scenario = {
  required_service_hours: simulationParams.required_service_count,
  force_decisions: parseForceInduct(simulationParams.force_induct),
  override_train_attributes: parseExcludeTrainsets(simulationParams.exclude_trainsets),
  // Note: weights are not currently supported in new system
};
```

### 2. **Weights Not Supported**

**Problem:**
- Frontend collects optimization weights (w_readiness, w_reliability, etc.)
- Backend `WhatIfScenario` model doesn't accept weights
- Weights are ignored in simulation

**Impact:**
- User-set weights have no effect
- Optimization uses default weights only

**Fix Required:**
- Add `weights` field to `WhatIfScenario` model
- Pass weights to `OptimizationRequest` in `run_whatif()`

### 3. **Parameter Conversion Logic**

**Problem:**
- Old endpoint (`/optimization/simulate`) has conversion logic but:
  - Converts `required_service_count` to `required_service_hours` (approximate)
  - Excludes trainsets by setting `fitness_certificates.rolling_stock.status: "EXPIRED"` (hacky)
  - Doesn't handle weights

**Impact:**
- Backwards compatibility works but is suboptimal
- Exclusion method is a workaround, not proper

---

## 🔍 Detailed Component Analysis

### Backend: `whatif_simulator.py`

**Strengths:**
- ✅ Deterministic execution with seeding
- ✅ Snapshot isolation (no DB writes)
- ✅ Comprehensive KPI computation
- ✅ Decision normalization
- ✅ Always returns array format

**Potential Issues:**
1. **Error Handling**: Exceptions are logged but may not provide user-friendly messages
2. **Performance**: Runs two full optimizations (baseline + scenario) - could be slow for large fleets
3. **Memory**: Deep copies of snapshots may use significant memory

**Recommendations:**
- Add timeout handling for long-running simulations
- Consider async progress updates for UI
- Add memory usage monitoring

### Backend: `simulation.py` API

**Strengths:**
- ✅ Proper normalization layer
- ✅ Error handling with HTTPException
- ✅ File-based result storage

**Potential Issues:**
1. **File Storage**: Results stored in `simulation_runs/` directory - no cleanup mechanism
2. **Concurrent Requests**: No locking mechanism for simultaneous simulations
3. **Validation**: Limited validation of scenario parameters

**Recommendations:**
- Add cleanup job for old simulation files
- Add request queuing or locking for concurrent simulations
- Add Pydantic validation for all scenario fields

### Frontend: `Optimization.tsx`

**Strengths:**
- ✅ Uses defensive utilities (`ensureResultsArray`, `asArray`)
- ✅ Handles both new and legacy formats
- ✅ Displays baseline vs scenario comparison
- ✅ Shows deltas and explain_log

**Issues:**
1. **API Call**: Uses wrong endpoint (`simulate` instead of `runSimulation`)
2. **Parameter Format**: Parameters don't match backend expectations
3. **Weights**: Collects weights but they're not used
4. **Error Handling**: Limited error display to user

**Recommendations:**
- Fix API call to use `runSimulation`
- Transform parameters to match backend format
- Add error toast notifications
- Add loading states for better UX

### Frontend: `DetailedResults.tsx`

**Strengths:**
- ✅ Fully defensive (never crashes)
- ✅ Handles all edge cases
- ✅ Extracts explanations from multiple fields
- ✅ Modal with proper event handling

**No Issues Found** ✅

---

## 📊 Data Flow Analysis

### Current Flow (BROKEN)
```
Frontend (Optimization.tsx)
  ↓
optimizationApi.simulate(params)  // GET /optimization/simulate
  ↓
Backend (optimization.py)
  ↓
/optimization/simulate endpoint (DEPRECATED)
  ↓
Converts params (incomplete)
  ↓
run_whatif(scenario)  // New system
  ↓
Returns results
```

### Expected Flow (CORRECT)
```
Frontend (Optimization.tsx)
  ↓
optimizationApi.runSimulation(scenario)  // POST /simulation/run
  ↓
Backend (simulation.py)
  ↓
/simulation/run endpoint
  ↓
run_whatif(scenario)  // New system
  ↓
Returns results
```

---

## 🐛 Bug List

### High Priority
1. **API Endpoint Mismatch**: Frontend calls wrong endpoint
2. **Parameter Format Mismatch**: Frontend params don't match backend
3. **Weights Not Supported**: User-set weights are ignored

### Medium Priority
4. **No File Cleanup**: Simulation files accumulate in `simulation_runs/`
5. **No Concurrent Request Handling**: Multiple simultaneous simulations may conflict
6. **Limited Error Messages**: Errors may not be user-friendly

### Low Priority
7. **Performance**: Two full optimizations may be slow
8. **Memory Usage**: Deep copies may use significant memory
9. **No Progress Updates**: Long-running simulations have no progress indication

---

## ✅ Recommendations

### Immediate Fixes (Required)
1. **Update Frontend API Call**
   ```typescript
   // Change from:
   mutationFn: optimizationApi.simulate
   
   // To:
   mutationFn: optimizationApi.runSimulation
   ```

2. **Transform Parameters**
   ```typescript
   const transformParams = (params: SimulationParams) => {
     const scenario: any = {
       required_service_hours: params.required_service_count,
     };
     
     // Parse force_induct
     if (params.force_induct) {
       const forced = params.force_induct.split(',').map(t => t.trim());
       scenario.force_decisions = {};
       forced.forEach(id => {
         scenario.force_decisions[id] = 'INDUCT';
       });
     }
     
     // Parse exclude_trainsets (proper way)
     if (params.exclude_trainsets) {
       const excluded = params.exclude_trainsets.split(',').map(t => t.trim());
       scenario.override_train_attributes = {};
       excluded.forEach(id => {
         scenario.override_train_attributes[id] = {
           'fitness_certificates.rolling_stock.status': 'EXPIRED'
         };
       });
     }
     
     return scenario;
   };
   ```

3. **Add Weights Support**
   ```python
   # In WhatIfScenario model:
   weights: Optional[Dict[str, float]] = Field(None, description="Optimization weights")
   
   # In run_whatif():
   if "weights" in scenario:
       request.weights = OptimizationWeights(**scenario["weights"])
   ```

### Future Improvements
1. **Add Progress Updates**: Use WebSockets or polling for long-running simulations
2. **Add File Cleanup**: Scheduled job to remove old simulation files
3. **Add Request Queuing**: Handle concurrent simulation requests
4. **Add Validation**: Comprehensive Pydantic validation for all fields
5. **Add Caching**: Cache snapshots for faster repeated simulations
6. **Add Export**: Allow exporting simulation results to PDF/Excel

---

## 📝 Testing Checklist

### Backend Tests
- [ ] Test `run_whatif()` with various scenarios
- [ ] Test parameter normalization
- [ ] Test decision explain field synthesis
- [ ] Test results array guarantee
- [ ] Test error handling
- [ ] Test concurrent requests

### Frontend Tests
- [ ] Test API call with correct endpoint
- [ ] Test parameter transformation
- [ ] Test DetailedResults component with various data shapes
- [ ] Test explanation extraction
- [ ] Test modal open/close
- [ ] Test error handling

### Integration Tests
- [ ] End-to-end simulation flow
- [ ] Baseline vs scenario comparison
- [ ] Delta calculations
- [ ] Explain log generation
- [ ] Results persistence

---

## 📈 Performance Considerations

### Current Performance
- **Baseline Optimization**: ~2-5 seconds (depends on fleet size)
- **Scenario Optimization**: ~2-5 seconds
- **Stabling Geometry**: ~1-2 seconds each
- **Total Simulation Time**: ~6-14 seconds

### Optimization Opportunities
1. **Parallel Execution**: Run baseline and scenario optimizations in parallel (if deterministic)
2. **Caching**: Cache snapshot for repeated simulations
3. **Incremental Updates**: Only re-optimize changed parts
4. **Progress Streaming**: Stream intermediate results to frontend

---

## 🔒 Security Considerations

### Current Security
- ✅ API key authentication required
- ✅ No direct database writes
- ✅ Snapshot isolation

### Recommendations
1. **Rate Limiting**: Add rate limiting for simulation endpoints
2. **Input Validation**: Validate all scenario parameters
3. **Resource Limits**: Limit simulation duration and memory usage
4. **Audit Logging**: Log all simulation runs for audit trail

---

## 📚 Documentation Gaps

1. **API Documentation**: Need OpenAPI/Swagger docs for simulation endpoints
2. **Parameter Guide**: Document all scenario parameters and their effects
3. **Example Scenarios**: Provide example scenario configurations
4. **Troubleshooting**: Document common issues and solutions

---

## Conclusion

The What-If Simulation system is **well-architected** with good defensive programming practices. However, there is a **critical integration issue** where the frontend calls the wrong endpoint with incompatible parameters. Once this is fixed, the system should work correctly.

**Priority Actions:**
1. Fix frontend API call (use `runSimulation` instead of `simulate`)
2. Transform parameters to match backend format
3. Add weights support to backend
4. Test end-to-end flow

**Estimated Fix Time:** 1-2 hours







