# What Does "Run AI/ML Optimization" Button Actually Do?

## Complete Step-by-Step Process

When you click **"Run AI/ML Optimization"** in the Optimization tab, here's exactly what happens:

---

## Phase 1: Data Collection & Preparation

### Step 1: Load All Trainsets from Database
```python
# Fetches all 25 trainsets from MongoDB
trainsets_data = await collection.find({})
```
- Retrieves current state of all trainsets
- Includes: certificates, job cards, mileage, branding, location, etc.

### Step 2: Calculate Required Train Count
```python
fleet_req = compute_required_trains(
    service_date=request.service_date,
    override_count=request.required_service_count
)
# Result: required_service_trains = 13, standby_buffer = 2
```
- Uses timetable-driven calculation
- Determines: **13 trains needed for service** (fixed by timetable)

---

## Phase 2: AI/ML Optimization Engine

### Step 3: Data Normalization & Type Safety
- Normalizes all trainset data to prevent type errors
- Ensures job cards are integers, mileage is float, etc.
- **Safety check**: Prevents bugs from type mismatches

### Step 4: ML Risk Prediction (AI Component)
```python
predictions = await batch_predict(features_for_pred)
```
- **Runs ML model** to predict failure risk for each train
- Calculates `predicted_failure_risk` (0.0 to 1.0)
- Calculates `ml_health_score` (0.0 to 1.0)
- Uses **SHAP values** for explainability
- **Deterministic**: Same input → same output (seeded)

### Step 5: Tier 1 - Safety Filtering (Hard Constraints)
**Filters out unsafe trains** - Cannot be overridden:

For each of 25 trainsets:
- ✅ **Fitness Certificates**: All 3 required (Rolling-Stock, Signalling, Telecom) must be valid
- ✅ **Job Cards**: No critical job cards open
- ✅ **Mileage**: Within maintenance limits
- ✅ **Maintenance Status**: Not currently in maintenance
- ✅ **Cleaning Slots**: Available if cleaning required

**Result**: 
- Example: 6 trains excluded (T-004, T-021, T-022, T-023, T-024, T-025)
- **19 trains eligible** for service

### Step 6: Tier 2 - High Priority Scoring (Revenue & Readiness)
**Scores each eligible train** for revenue and operational readiness:

- **Branding Obligation**: +300 points if train has active advertiser wrap
  - HIGH priority: +300 points
  - MEDIUM priority: +180 points (60%)
  - LOW priority: +90 points (30%)
- **Minor Defects**: -50 points per open job card (non-critical)

**Purpose**: Prioritize trains that generate revenue and are operationally ready

### Step 7: Tier 3 - Optimization Scoring (Fleet Health)
**Scores each train** for long-term fleet optimization:

- **Mileage Balancing**: +50 points if below average (equalize wear)
- **Cleaning Due**: -30 points if cleaning required soon
- **Shunting Complexity**: -20 points if train is blocked/requires movement
- **ML Health Score**: +100 × health_score (better health = higher score)

**Purpose**: Optimize for fleet longevity, cost reduction, operational efficiency

### Step 8: OR-Tools Solver (Mathematical Optimization)
```python
solver = pywraplp.Solver.CreateSolver("SCIP")
solver.Maximize(combined_score)
solver.Add(sum(selected_trains) == 13)  # Exactly 13 trains
```

**What it does**:
- Creates optimization problem: "Select exactly 13 trains from 19 eligible"
- Maximizes combined score (Tier 2 dominates Tier 3)
- Uses **linear programming** to find optimal solution
- **Result**: Best 13 trains selected for service

**If solver unavailable**: Falls back to tiered scoring (sorts by score, picks top 13)

### Step 9: Decision Assignment
For each of 25 trainsets, assigns one of three decisions:

- **INDUCT** (13 trains): Selected for revenue service
- **STANDBY** (6 trains): Available but not needed
- **MAINTENANCE** (6 trains): Critical failures or maintenance required

### Step 10: Explainability Generation
For each decision, generates:
- **Top Reasons**: Why this train was selected/excluded
- **Top Risks**: Potential issues to watch
- **SHAP Values**: ML feature importance
- **Confidence Score**: How certain the decision is
- **Composite Score**: Overall optimization score

---

## Phase 3: Stabling Geometry Optimization

### Step 11: Bay Assignment Optimization
```python
stabling_geometry = await stabling_optimizer.optimize_stabling_geometry(
    trainsets_data, decisions
)
```

**What it does**:
- Groups trains by depot (Aluva, Petta)
- Assigns physical bay positions to minimize:
  - **Shunting time**: Nightly movements
  - **Turnout time**: Morning dispatch time
- **Prevents conflicts**: No two trains in same bay
- **Capacity checks**: Warns if capacity exceeded

**Result**:
- Bay assignments for each train
- Shunting schedule (which trains to move)
- Total shunting time
- Total turnout time
- Efficiency metrics

---

## Phase 4: Data Persistence & Response

### Step 12: Store Results in Database
```python
await store_optimization_history(request, optimization_result)
```

**Saves to MongoDB**:
- Optimization history (for audit trail)
- Latest induction list (for quick access)
- Decisions with full explainability

### Step 13: Generate Response
Returns comprehensive results:

```json
{
  "required_service_trains": 13,
  "standby_buffer": 2,
  "total_required_trains": 15,
  "calculation_method": "timetable",
  "eligible_train_count": 19,
  "granted_train_count": 13,
  "actual_induct_count": 13,
  "service_shortfall": 0,
  "decisions": [
    {
      "trainset_id": "T-001",
      "decision": "INDUCT",
      "score": 0.95,
      "confidence_score": 0.92,
      "top_reasons": ["Valid certificates", "High branding priority", "Low mileage"],
      "top_risks": [],
      "shap_values": [...]
    },
    // ... 24 more decisions
  ],
  "stabling_geometry": {
    "optimized_layout": {
      "Aluva": {
        "bay_assignments": {"T-001": 6, "T-002": 7, ...},
        "shunting_operations": [...],
        "total_shunting_time": 45,
        "total_turnout_time": 78
      }
    },
    "capacity_warning": false,
    "efficiency_improvement": 15.5
  }
}
```

---

## What Makes This "AI/ML"?

### 1. **Machine Learning Risk Prediction**
- Uses trained PyTorch model to predict failure risk
- Learns from historical data
- Provides SHAP explainability

### 2. **Mathematical Optimization (OR-Tools)**
- Solves complex multi-objective optimization problem
- Considers 6 interdependent variables simultaneously
- Finds optimal solution (not just "good enough")

### 3. **Tiered Constraint Hierarchy**
- Lexicographic optimization (safety > revenue > efficiency)
- Handles conflicting objectives intelligently
- Ensures safety constraints are never violated

---

## Real-World Impact

### Before Optimization (Manual Process):
- ⏱️ **Time**: 2-3 hours of manual work
- ❌ **Errors**: Missed expired certificates, ignored branding
- ❌ **Inconsistency**: Different criteria each night
- ❌ **No Explainability**: Can't justify decisions

### After Optimization (AI-Driven):
- ⚡ **Time**: Seconds (automated)
- ✅ **Safety**: All unsafe trains automatically excluded
- ✅ **Revenue**: Branding obligations prioritized
- ✅ **Efficiency**: Mileage balanced, shunting minimized
- ✅ **Consistency**: Same criteria every night
- ✅ **Explainability**: Clear reasoning for every decision
- ✅ **Auditability**: Complete trail in database

---

## Key Outputs

1. **Ranked List**: All 25 trainsets with decisions (INDUCT/STANDBY/MAINTENANCE)
2. **Bay Assignments**: Physical positions in depots
3. **Shunting Schedule**: Which trains to move and when
4. **Efficiency Metrics**: Time savings, capacity utilization
5. **Explainability**: Why each decision was made
6. **Risk Assessment**: ML predictions for each train

---

## When to Run It

**Recommended**: Run every night between 21:00-23:00 IST
- Before nightly operations begin
- After all data sources updated (Maximo, sensors, etc.)
- Gives operations team time to review and adjust

**Can also run**:
- After data updates (new job cards, certificate renewals)
- For "what-if" scenarios
- To validate manual overrides

---

## Summary

The "Run AI/ML Optimization" button:

1. ✅ **Calculates** how many trains needed (13 from timetable)
2. ✅ **Evaluates** all 25 trainsets using AI/ML
3. ✅ **Selects** the best 13 trains for service
4. ✅ **Assigns** remaining trains to STANDBY/MAINTENANCE
5. ✅ **Optimizes** bay positions to minimize shunting
6. ✅ **Explains** every decision with reasoning
7. ✅ **Stores** results for audit and review

**It transforms a 2-3 hour manual process into a seconds-long automated decision with better outcomes.**

---

**Last Updated**: 2025-01

