# Multi-Depot Simulation Hotfix - Quick Checklist

## Pre-Implementation
- [ ] Read `HOTFIX_IMPLEMENTATION_GUIDE.md` completely
- [ ] Backup current code: `git stash` or create feature branch
- [ ] Ensure tests can run: `cd backend && python -m pytest tests/ -v`

## Implementation Tasks (Do in Order)

### Phase 1: Backend Core (Critical)
- [ ] **A**: Fix response schema keys in `coordinator.py` and `multi_depot_simulate.py`
- [ ] **B**: Fix shunting/turnout aggregation in `coordinator.py`
- [ ] **C**: Add `used_ai` tracking in `multi_depot_simulate.py` and `coordinator.py`
- [ ] **D**: Add error handling with JSON responses in `multi_depot_simulate.py`
- [ ] Test: Run API smoke test (see section L)

### Phase 2: Frontend Mapping
- [ ] **F**: Update frontend to use correct response keys in `MultiDepotSimulation.tsx`
- [ ] **E**: Add Required Service and Used AI cards in `MultiDepotSimulation.tsx`
- [ ] Test: Check UI shows correct values

### Phase 3: UX Improvements
- [ ] **G**: Add pre-run validation modal in `MultiDepotSimulation.tsx`
- [ ] **H**: Add terminal overflow distribution in `coordinator.py`
- [ ] **I**: Make transfer planner conditional in `coordinator.py`
- [ ] Test: Run full UI flow (see section L)

### Phase 4: Observability
- [ ] **J**: Add logging in `main.py` and `multi_depot_simulate.py`
- [ ] **K**: Create test files and add tests
- [ ] Test: Run `pytest tests/test_simulation_response_schema.py -v`

### Phase 5: Final Polish
- [ ] **N**: Add judge note in UI
- [ ] **M**: Write commit messages and update README
- [ ] **L**: Run full manual QA checklist

## Verification Steps

### After Phase 1:
```bash
# Test API
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"fleet": 5, "depots": [{"name": "Muttom", "service_bays": 6, "maintenance_bays": 4, "standby_bays": 2, "location_type": "FULL_DEPOT"}], "ai_mode": false}'

# Check response has: service_trains, shunting_time, used_ai
```

### After Phase 2:
- Open UI: `http://localhost:5173/multi-depot-simulation`
- Run small simulation (fleet=5, 1 depot)
- Verify: shunting_time > 0, required_service visible, used_ai badge shows

### After Phase 3:
- Configure fleet=40, Muttom only
- Verify: Overflow modal appears
- Click "Add Terminal Presets"
- Verify: Aluva/Petta added
- Run simulation
- Verify: Results show reduced shortfall or terminals used

### After Phase 4:
```bash
# Check logs
tail -f logs/app.log | grep SIMULATE_RUN

# Run tests
cd backend && python -m pytest tests/test_simulation_response_schema.py -v
```

## Commit Strategy

Commit after each phase:
1. `git add .`
2. `git commit -m "fix(sim): [phase description]"`
3. Test before next phase

## Rollback

If issues:
```bash
git log --oneline  # Find commit hash
git revert <commit-hash>
```

## Success Indicators

- ✅ UI shows non-zero shunting_time
- ✅ Required service visible
- ✅ Used AI badge accurate
- ✅ Overflow modal works
- ✅ Tests pass
- ✅ Logs structured

---

**Estimated Time**: 2-3 hours for full implementation
**Priority**: High (fixes critical UI display issues)

