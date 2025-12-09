# Verify Multi-Depot Simulation Setup

## ✅ Quick Verification Checklist

### 1. Files Exist?
```bash
# Frontend
frontend/src/pages/MultiDepotSimulation.tsx ✓
frontend/src/App.tsx (has import and route) ✓
frontend/src/components/layout/AppSidebar.tsx (has link) ✓
frontend/src/services/api.ts (has multiDepotSimulationApi) ✓

# Backend
backend/app/api/multi_depot_simulate.py ✓
backend/app/services/simulation/coordinator.py ✓
backend/app/config/depots.yaml ✓
```

### 2. Route is Registered?
Open `frontend/src/App.tsx` and verify:
- Line 13: `import MultiDepotSimulation from "./pages/MultiDepotSimulation";`
- Line 124-130: Route definition exists

### 3. Sidebar Link Added?
Open `frontend/src/components/layout/AppSidebar.tsx` and verify:
- Line 83-89: "Multi-Depot Simulation" link exists

### 4. API Service Added?
Open `frontend/src/services/api.ts` and verify:
- `multiDepotSimulationApi` object exists at end of file

## 🚀 How to See Changes

### Step 1: Restart Dev Server
```bash
# Stop current server (Ctrl+C)
cd frontend
npm run dev
```

### Step 2: Hard Refresh Browser
- **Windows**: `Ctrl + Shift + R` or `Ctrl + F5`
- **Mac**: `Cmd + Shift + R`

### Step 3: Access the Page

**Option A: Via Sidebar**
1. Login as ADMIN or OPERATIONS_MANAGER
2. Look in sidebar for "Multi-Depot Simulation" (🏢 icon)
3. Click it

**Option B: Direct URL**
```
http://localhost:5173/multi-depot-simulation
```

### Step 4: Verify It Works
You should see:
- Title: "Multi-Depot Simulation"
- Fleet size buttons (25, 40, 60, 100)
- Depot configuration section
- "Run Simulation" button

## 🔍 Troubleshooting

### If sidebar link doesn't appear:
1. **Check user role**: Must be ADMIN or OPERATIONS_MANAGER
2. **Logout and login again**
3. **Check browser console** (F12) for errors

### If page shows 404:
1. **Verify route in App.tsx** (should be at line ~124)
2. **Check import statement** (line 13)
3. **Restart dev server**

### If page loads but API fails:
1. **Start backend server**:
   ```bash
   cd backend
   python start_dev.py
   ```
2. **Test API directly**:
   ```
   http://localhost:8000/api/v1/depots/presets
   ```

### If nothing appears:
1. **Clear browser cache completely**
2. **Close and reopen browser**
3. **Check terminal for compilation errors**
4. **Verify all files exist** (use checklist above)

## 📝 Quick Test

1. Navigate to: `http://localhost:5173/multi-depot-simulation`
2. Click "40 trains, 2 depots" button
3. Click "Run Simulation"
4. Wait for results
5. Check "Global Summary" tab

If this works, everything is set up correctly!

## 🐛 Still Not Working?

Run this command to check for errors:
```bash
cd frontend
npm run typecheck 2>&1 | grep -i "MultiDepotSimulation"
```

If no errors, the component is fine. The issue is likely:
- Browser cache (hard refresh)
- Dev server not restarted
- Wrong user role

