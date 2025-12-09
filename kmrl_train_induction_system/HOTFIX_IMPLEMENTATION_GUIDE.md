# Multi-Depot Simulation UI Hotfix - Implementation Guide

## Overview
This guide provides step-by-step instructions to fix UI display issues, add AI/fallback visibility, and ensure proper response schema alignment. All changes are minimal, targeted, and reversible.

---

## A. BACKEND: RESPONSE SCHEMA ALIGNMENT (HIGH PRIORITY)

### Files to Edit:
1. `backend/app/services/simulation/coordinator.py` - Function `_compute_global_summary()`
2. `backend/app/api/multi_depot_simulate.py` - Response wrapper in `simulate_multi_depot()`

### What to Change:

**In `coordinator.py` - `_compute_global_summary()` function (around line 206-243):**

1. **Rename existing keys to match UI expectations:**
   - Change `total_shunting_time_min` → `shunting_time` (keep value in minutes)
   - Change `total_turnout_time_min` → `turnout_time` (keep value in minutes)
   - Change `total_service_trains` → `service_trains`
   - Change `required_service_trains` → `required_service`
   - Change `effective_service_shortfall` → `service_shortfall`
   - Change `total_transfers_recommended` → `transfers_recommended`

2. **Add missing fields:**
   - Add `total_capacity`: Sum of all depot `total_bays` from depot configs
   - Add `fleet`: Pass through the `fleet_count` parameter
   - Add `used_ai`: This will be set in the API layer (see section C)

3. **Ensure all values are integers:**
   - Wrap any float calculations in `int()` to ensure integer output

### In `multi_depot_simulate.py` - `simulate_multi_depot()` function (around line 103-177):

1. **Add `used_ai` tracking:**
   - Before calling `run_simulation()`, check if `request.ai_mode` is True
   - After simulation, determine if AI was actually used (see section C)
   - Add `used_ai` boolean to the response at the same level as `run_id`

2. **Ensure `global_summary` contains all required keys:**
   - After getting `result.global_summary`, add a small adapter function that maps/validates keys
   - If any key is missing, set a default value (0 for numbers, empty list for arrays)
   - Add a warning entry if defaults were used: "Some summary values were defaulted; check logs"

### Why:
The UI expects specific key names. Current backend uses different names (e.g., `total_shunting_time_min` vs `shunting_time`), causing UI to show 0. This alignment fixes the mismatch.

---

## B. BACKEND: ENSURE SHUNTING & TURNOUT AGGREGATION

### Files to Edit:
1. `backend/app/services/simulation/coordinator.py` - `_compute_global_summary()`
2. `backend/app/services/simulation/depot_simulator.py` - `_compute_shunting_summary()`

### What to Change:

**In `coordinator.py` - `_compute_global_summary()` function:**

1. **Fix shunting time aggregation:**
   - Current code sums `result.shunting_summary.get("total_time_min", 0)` - this is correct
   - Ensure the sum is stored as `shunting_time` (after renaming from section A)

2. **Fix turnout time calculation:**
   - Current code uses `result.kpis.get("shunting_time_min", 0) * 0.8`
   - Change to use `result.shunting_summary.get("total_time_min", 0) * 0.8` for consistency
   - Store as `turnout_time` (after renaming from section A)

3. **Add shunting feasibility check:**
   - After aggregating times, check if all depots have `shunting_summary.get("feasible", False) == True`
   - Add `shunting_feasible: bool` to global_summary (True only if all depots feasible)

**In `depot_simulator.py` - `_compute_shunting_summary()` function (around line 183-195):**

1. **Ensure `total_time_min` is always present:**
   - If `shunting_ops` is empty, return `total_time_min: 0`
   - Ensure `feasible` boolean is always set (True if total_time <= max_shunting_window_min)

### Why:
UI expects global totals. Current aggregation may miss values if keys don't match or if some depots don't have shunting operations.

---

## C. BACKEND: ADD USED_AI / FALLBACK VISIBILITY

### Files to Edit:
1. `backend/app/api/multi_depot_simulate.py` - `simulate_multi_depot()` function
2. `backend/app/services/simulation/coordinator.py` - `run_simulation()` function

### What to Change:

**In `multi_depot_simulate.py` - `simulate_multi_depot()` function:**

1. **Track AI usage:**
   - Before calling `run_simulation()`, set a local variable `actually_used_ai = False`
   - If `request.ai_mode` is True, check if AI services are available (see below)
   - After `run_simulation()` completes, set `actually_used_ai` based on whether AI was actually used
   - Add `"used_ai": actually_used_ai` to the response JSON (top level, not in global_summary)

2. **Add fallback warning:**
   - If `request.ai_mode` is True but `actually_used_ai` is False, append to `result.warnings`: "AI services unavailable; deterministic fallback used"

**In `coordinator.py` - `run_simulation()` function:**

1. **Return AI usage status:**
   - Add a return value or modify SimulationResult to include `used_ai: bool`
   - If `ai_mode=True` but AI services fail/unavailable, set `used_ai=False`
   - If `ai_mode=False`, set `used_ai=False`
   - If `ai_mode=True` and AI services work, set `used_ai=True`

2. **Simple AI health check (if not exists):**
   - Create a small function `_check_ai_services_available()` that returns True/False
   - For now, if `ai_mode=True`, assume AI is available (we'll add real health check later)
   - This can be a stub that always returns True for now, but structure it so it can be enhanced

### Why:
Judges need to know if results came from AI or deterministic fallback. Transparency prevents confusion about result quality.

---

## D. BACKEND: ROUTE - ALWAYS RETURN JSON WITH RUN_ID & WARNINGS

### Files to Edit:
1. `backend/app/api/multi_depot_simulate.py` - `simulate_multi_depot()` function

### What to Change:

**In `multi_depot_simulate.py` - `simulate_multi_depot()` function:**

1. **Wrap entire function in try-except:**
   - Wrap the simulation logic (from line ~103 to ~179) in a try-except block
   - In the except block, catch `Exception as e`
   - Log the full stacktrace using `logger.exception("Simulation failed")`
   - Generate a run_id even on error: `run_id = f"error_{uuid.uuid4().hex[:8]}"`
   - Return HTTP 500 with JSON: `{"message": "Simulation failed", "run_id": run_id, "short_error": str(e)[:200]}`
   - Write error to file: Append to `logs/simulation_errors.log` with format: `{run_id}|{timestamp}|{short_error}`

2. **Ensure response always has run_id:**
   - Before returning success response, verify `result.run_id` exists
   - If missing, generate one: `run_id = _generate_run_id(...)` or use UUID

3. **Ensure warnings is always a list:**
   - If `result.warnings` is None or missing, set to empty list `[]`

### Why:
Raw 500 HTML pages are not actionable. JSON errors with run_id allow operators to report issues and check logs.

---

## E. FRONTEND: SHOW REQUIRED SERVICE & USED_AI PROMINENTLY

### Files to Edit:
1. `frontend/src/pages/MultiDepotSimulation.tsx` - Results display section

### What to Change:

**In `MultiDepotSimulation.tsx` - Results section (around line 300-400):**

1. **Add Required Service card:**
   - In the Global Summary area (where other summary tiles are shown)
   - Add a new card/tile component showing: "Required Service: {value}"
   - If the value was computed (not user-provided), show a small info icon with tooltip: "Computed from fleet size; click to override"
   - Style it similarly to other summary tiles (service_trains, shunting_time, etc.)

2. **Add Used AI badge:**
   - Add a prominent badge/indicator showing "Used AI: Yes" or "Used AI: No"
   - Color: Green if Yes, Yellow/Orange if No
   - Add tooltip: "AI used: shows whether ML inference was used; if No, result used deterministic fallback"
   - Place it near the run_id or at the top of results section

3. **Display warnings prominently:**
   - Ensure warnings list is visible (not collapsed)
   - If `used_ai === false`, ensure the fallback warning appears in the warnings list
   - Style warnings with appropriate icons (AlertTriangle for warnings, XCircle for errors)

### Why:
Operators need to see at a glance what was required and whether AI was used. Hiding this information causes confusion.

---

## F. FRONTEND: MAP RESPONSE KEYS TO UI (SCHEMA CHECK)

### Files to Edit:
1. `frontend/src/pages/MultiDepotSimulation.tsx` - State management and result parsing
2. `frontend/src/services/api.ts` - API response handling (if exists)

### What to Change:

**In `MultiDepotSimulation.tsx` - Result handling:**

1. **Update state interface:**
   - Find the interface/type definition for `SimulationResult` or `simulationResult` state
   - Ensure it includes: `global_summary: { service_trains, shunting_time, turnout_time, total_capacity, required_service, service_shortfall, used_ai, warnings, ... }`

2. **Add defensive parsing:**
   - After receiving API response, add a function `normalizeSimulationResponse(data)`
   - This function checks if `data.global_summary` has all expected keys
   - If a key is missing, set default: `service_trains: 0, shunting_time: 0, turnout_time: 0, etc.`
   - If defaults were used, add a UI warning: "Some values missing from response; using defaults"

3. **Update display code:**
   - Find where `simulationResult.global_summary.service_trains` is accessed
   - Ensure it uses the new key names (not `total_service_trains`, etc.)
   - Add optional chaining: `simulationResult?.global_summary?.service_trains ?? 0`

**In `api.ts` (if exists):**

1. **Add response transformer:**
   - In the `multiDepotSimulationApi.simulate()` function, add a response transformer
   - Map old key names to new ones if backend still returns old names (backward compatibility)
   - Ensure `used_ai` is extracted from top-level response, not nested

### Why:
Frontend may be reading wrong keys or missing keys, causing 0 values. Defensive parsing prevents crashes and shows warnings when data is incomplete.

---

## G. FRONTEND: PRE-RUN VALIDATION & SAFE PROMPT

### Files to Edit:
1. `frontend/src/pages/MultiDepotSimulation.tsx` - `handleRunSimulation()` function

### What to Change:

**In `MultiDepotSimulation.tsx` - Before simulation:**

1. **Add capacity calculation:**
   - In `handleRunSimulation()` function, before calling the API
   - Calculate `total_capacity = depots.reduce((sum, d) => sum + d.service_bays + d.maintenance_bays + d.standby_bays, 0)`
   - Add estimated terminal capacity: If no terminals (TERMINAL_YARD) present, add 12 (6 for Aluva + 6 for Petta)

2. **Add overflow check:**
   - If `fleetSize > total_capacity + estimated_terminal_capacity`:
     - Show a modal/dialog with message: "Fleet (X) exceeds total capacity (Y) by Z trains. Options:"
     - Add three buttons:
       - "Add Terminal Presets" - Auto-adds Aluva and Petta terminal presets to depots list
       - "Reduce Fleet" - Sets fleetSize to total_capacity
       - "Proceed Anyway" - Closes modal and continues with simulation
   - Use a Dialog component from shadcn/ui or similar

3. **Auto-add terminals:**
   - Create a function `addTerminalPresets()` that adds Aluva and Petta to depots
   - Aluva: `{ name: "Aluva Terminal", location_type: "TERMINAL_YARD", service_bays: 0, maintenance_bays: 0, standby_bays: 6 }`
   - Petta: `{ name: "Petta Terminal", location_type: "TERMINAL_YARD", service_bays: 0, maintenance_bays: 0, standby_bays: 6 }`

### Why:
Prevents accidental runs that obviously overflow. Gives operators clear options to fix the issue before running.

---

## H. BACKEND: DISTRIBUTE OVERFLOW TO TERMINALS (DETERMINISTIC FALLBACK)

### Files to Edit:
1. `backend/app/services/simulation/coordinator.py` - `_partition_fleet()` function
2. `backend/app/services/simulation/depot_simulator.py` - `simulate_depot()` function

### What to Change:

**In `coordinator.py` - `_partition_fleet()` function (around line 130-203):**

1. **Identify terminals:**
   - After partitioning fleet to depots, check if any depot has `location_type == LocationType.TERMINAL_YARD`
   - Sort terminals by config order or proximity (use `display_order` if available)

2. **Distribute overflow:**
   - Calculate total capacity: Sum of all depot `total_bays`
   - If `fleet_count > total_capacity`:
     - Calculate overflow: `overflow = fleet_count - total_capacity`
     - For each terminal (in order), assign overflow trains to `standby_bays` until terminal is full or overflow is exhausted
     - Track which trains went to terminals in assignment map

3. **Add overflow warning:**
   - If overflow exists and terminals were used, add warning: "X trains placed in terminal standby bays due to capacity overflow"
   - If overflow exists but no terminals, add warning: "X trains exceed capacity; consider adding terminal presets (Aluva/Petta)"

**In `depot_simulator.py` - `simulate_depot()` function:**

1. **Handle terminal assignments:**
   - If depot is TERMINAL_YARD and has assigned trains, place them in standby_bays only
   - Don't try to assign to service_bays or maintenance_bays (they're 0 for terminals)

### Why:
Realistic behavior: terminals can hold overflow trains even without full workshop capacity. This makes simulations more accurate for Kochi metro.

---

## I. BACKEND: TRANSFER PLANNER TRIGGER

### Files to Edit:
1. `backend/app/services/simulation/coordinator.py` - `run_simulation()` function

### What to Change:

**In `coordinator.py` - `run_simulation()` function (around line 70-75):**

1. **Conditional transfer planning:**
   - Current code always calls `plan_transfers()`
   - Add condition: Only call if `len(depots) > 1` OR if any depot has `location_type == LocationType.TERMINAL_YARD`
   - If condition not met, set `transfer_recommendations = []`

2. **Ensure transfer count:**
   - After `plan_transfers()`, ensure the returned list is not None
   - Count recommended transfers: `transfers_recommended = len([t for t in transfer_recommendations if t.recommended])`
   - This count should already be in `global_summary.transfers_recommended` (from section A)

### Why:
Transfer planner should only run when it makes sense (multiple depots or terminals present). Avoids unnecessary computation.

---

## J. LOGGING & METRICS (small)

### Files to Edit:
1. `backend/app/main.py` - Logging configuration
2. `backend/app/api/multi_depot_simulate.py` - `simulate_multi_depot()` function

### What to Change:

**In `main.py` - Startup configuration:**

1. **Ensure logging setup:**
   - Check if `logging.basicConfig()` is called
   - If not, add: `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('logs/app.log'), logging.StreamHandler()])`
   - Ensure `logs/` directory exists (create if not)

**In `multi_depot_simulate.py` - `simulate_multi_depot()` function:**

1. **Add simulation metric log:**
   - After successful simulation, log: `logger.info(f"SIMULATE_RUN run_id={result.run_id} fleet={request.fleet} depots={len(depot_configs)} used_ai={actually_used_ai} duration={duration:.2f}")`
   - Calculate `duration` by recording start time before `run_simulation()` and end time after

2. **Create logs directory:**
   - At module level, ensure `logs/` directory exists: `Path('logs').mkdir(exist_ok=True)`

### Why:
Structured logs make postmortems easier and help judges understand system behavior during demos.

---

## K. TESTS (automated)

### Files to Create:
1. `backend/tests/test_simulation_response_schema.py`
2. `backend/tests/test_simulate_endpoint_fallback.py`

### What to Add:

**In `test_simulation_response_schema.py`:**

1. **Test response structure:**
   - Create test function `test_simulation_response_has_required_keys()`
   - POST to `/api/v1/simulate` with payload: `{fleet: 5, depots: [{name: "Muttom", service_bays: 6, maintenance_bays: 4, standby_bays: 2, location_type: "FULL_DEPOT"}], ai_mode: false}`
   - Assert HTTP 200
   - Assert `response.json()["global_summary"]` contains all keys: `service_trains`, `shunting_time`, `turnout_time`, `total_capacity`, `required_service`, `service_shortfall`, `used_ai`, `warnings`
   - Assert all values are correct types (int for numbers, list for warnings)

**In `test_simulate_endpoint_fallback.py`:**

1. **Test AI fallback:**
   - Mock AI health checker to return unhealthy
   - POST with `ai_mode: true`
   - Assert `response.json()["used_ai"] == false`
   - Assert `response.json()["warnings"]` contains string "deterministic fallback"

### Why:
Prevents regressions. Automated tests catch schema changes before they reach production.

---

## L. MANUAL QA CHECKLIST

### Steps to Verify:

1. **Restart services:**
   - Stop backend: `Ctrl+C` in backend terminal
   - Restart: `cd backend && uvicorn app.main:app --reload`
   - Frontend should auto-reload (Vite HMR)

2. **API smoke test:**
   - Use curl or Postman: `POST http://localhost:8000/api/v1/simulate`
   - Payload: `{"fleet": 5, "depots": [{"name": "Muttom", "service_bays": 6, "maintenance_bays": 4, "standby_bays": 2, "location_type": "FULL_DEPOT"}], "ai_mode": false}`
   - Verify: HTTP 200, `global_summary.service_trains > 0`, `global_summary.shunting_time > 0`, `used_ai: false`

3. **UI flow test:**
   - Open `http://localhost:5173/multi-depot-simulation`
   - Configure: Fleet 40, Muttom only (12 bays total)
   - Click "Run Simulation"
   - Verify: Modal appears warning about overflow
   - Click "Add Terminal Presets"
   - Verify: Aluva and Petta added (total capacity now 24)
   - Run simulation
   - Verify: Results show `service_shortfall` reduced, `transfers_recommended` > 0 or overflow placed in terminals

4. **Log verification:**
   - Check `logs/app.log` for `SIMULATE_RUN` entries
   - Check `logs/simulation_errors.log` is empty (or only contains expected test errors)

### Why:
Manual testing catches UI/UX issues that automated tests miss. Ensures end-to-end flow works for judges.

---

## M. COMMIT MESSAGES AND DOCUMENTATION

### Commit Messages (one per logical change):

1. `fix(sim): align simulation response schema and expose shunting_time/turnout_time`
   - Files: `coordinator.py`, `multi_depot_simulate.py`
   - Changes: Renamed keys in global_summary to match UI expectations

2. `feat(sim): add used_ai visibility and deterministic fallback warning`
   - Files: `coordinator.py`, `multi_depot_simulate.py`
   - Changes: Added used_ai boolean and fallback warnings

3. `fix(ui): map simulation response keys and add required_service card`
   - Files: `MultiDepotSimulation.tsx`
   - Changes: Added defensive parsing and UI cards for required_service/used_ai

4. `feat(ui): add pre-run validation and terminal preset suggestions`
   - Files: `MultiDepotSimulation.tsx`
   - Changes: Added overflow check modal and auto-add terminal presets

5. `feat(sim): distribute overflow to terminals and improve transfer planner`
   - Files: `coordinator.py`, `depot_simulator.py`
   - Changes: Overflow trains go to terminals; transfer planner conditional

6. `chore(log): add simulation metrics and error logging`
   - Files: `main.py`, `multi_depot_simulate.py`
   - Changes: Added structured logging and error file

7. `test(sim): add simulation response schema tests`
   - Files: `test_simulation_response_schema.py`, `test_simulate_endpoint_fallback.py`
   - Changes: Added automated tests for response structure and fallback

### README Update:

Add section to `README.md`:

```markdown
## Simulation: Response Schema and AI Fallback

The simulation API returns a standardized response with the following structure:

- `run_id`: Unique identifier for the simulation run
- `used_ai`: Boolean indicating whether AI/ML services were used (false = deterministic fallback)
- `global_summary`: Contains `service_trains`, `shunting_time`, `turnout_time`, `total_capacity`, `required_service`, `service_shortfall`, `transfers_recommended`, `warnings`
- `warnings`: List of warnings; includes "AI services unavailable; deterministic fallback used" if fallback was used

If AI services are unavailable and `ai_mode=true`, the system automatically falls back to deterministic heuristics. This is transparent to the user via the `used_ai` flag and warnings.
```

---

## N. DELIVERY NOTE TO JUDGES (final UI copy)

### Add to UI:

In `MultiDepotSimulation.tsx`, add a small info box at the top of Results section:

```tsx
<Alert className="mb-4">
  <Info className="h-4 w-4" />
  <AlertDescription>
    Note: Results may use AI inference or a deterministic fallback. 
    If fallback used, a warning is shown. Use "Used AI" badge to confirm.
  </AlertDescription>
</Alert>
```

### Why:
Sets expectations for judges. They'll know to check the badge and understand what it means.

---

## Implementation Order

1. **A** (Response Schema) - Do first, fixes core display issues
2. **B** (Shunting Aggregation) - Do with A, related
3. **C** (Used AI) - Do after A/B, depends on response structure
4. **D** (Error Handling) - Do early, prevents bad UX on errors
5. **F** (Frontend Mapping) - Do after A, frontend needs correct keys
6. **E** (UI Cards) - Do after F, depends on correct data
7. **G** (Pre-run Validation) - Do after E, improves UX
8. **H** (Terminal Overflow) - Do after G, related to overflow handling
9. **I** (Transfer Planner) - Do with H, related logic
10. **J** (Logging) - Do anytime, independent
11. **K** (Tests) - Do after A-C, validates changes
12. **L** (QA) - Do last, final verification
13. **M** (Commits) - Do as you go, one commit per logical change
14. **N** (UI Note) - Do with E, part of UI improvements

---

## Rollback Plan

If issues arise, each change is isolated:
- Revert commits one at a time
- Check `git log` for commit hashes
- Use `git revert <commit-hash>` to undo specific changes

---

## Success Criteria

After implementation:
- ✅ UI shows non-zero shunting_time and turnout_time
- ✅ Required service is visible and correct
- ✅ Used AI badge shows correct status
- ✅ Warnings appear when fallback used
- ✅ Overflow modal appears for large fleets
- ✅ Terminals receive overflow trains
- ✅ Tests pass
- ✅ Logs contain structured entries

---

End of Implementation Guide

