# Quick Fix: Multi-Depot Simulation Not Showing in Browser

## Immediate Steps

### 1. Hard Refresh Browser
- **Windows/Linux**: Press `Ctrl + Shift + R` or `Ctrl + F5`
- **Mac**: Press `Cmd + Shift + R`
- This clears cache and reloads all files

### 2. Check Dev Server
Make sure Vite dev server is running:
```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 3. Direct URL Access
Open directly in browser:
```
http://localhost:5173/multi-depot-simulation
```

### 4. Check Sidebar
- Look for "Multi-Depot Simulation" link in sidebar
- Should be between "Optimization" and "Data Ingestion"
- Icon: Building2 (🏢)

### 5. Check User Role
- Must be logged in as **ADMIN** or **OPERATIONS_MANAGER**
- If not, logout and login as admin

### 6. Restart Dev Server
If still not working:
```bash
# Stop server (Ctrl+C in terminal)
cd frontend
npm run dev
```

### 7. Check Browser Console
Press `F12` → Console tab:
- Look for red errors
- Check if route is loading
- Look for 404 errors

### 8. Verify Backend is Running
```bash
cd backend
python start_dev.py
```

Test API:
```
http://localhost:8000/api/v1/depots/presets
```

## What Changed

✅ **New Route Added**: `/multi-depot-simulation`
✅ **Sidebar Link Added**: "Multi-Depot Simulation" 
✅ **Component Created**: `MultiDepotSimulation.tsx`
✅ **API Endpoints**: `/api/v1/simulate`, `/api/v1/depots/presets`

## If Still Not Working

1. **Check file exists**:
   ```bash
   ls frontend/src/pages/MultiDepotSimulation.tsx
   ```

2. **Check route in App.tsx**:
   - Open `frontend/src/App.tsx`
   - Search for "multi-depot-simulation"
   - Should find route definition

3. **Check sidebar**:
   - Open `frontend/src/components/layout/AppSidebar.tsx`
   - Search for "Multi-Depot Simulation"
   - Should be in navigationItems array

4. **Clear node_modules and reinstall** (last resort):
   ```bash
   cd frontend
   rm -rf node_modules
   npm install
   npm run dev
   ```

## Expected Behavior

When you access the page, you should see:
1. **Title**: "Multi-Depot Simulation"
2. **Simulation Configuration Card** with:
   - Fleet size buttons (25, 40, 60, 100)
   - Depot configuration section
   - Add depot buttons
   - Run Simulation button
3. **Stress Test Presets Card**
4. **Results section** (after running simulation)

## Still Having Issues?

Check these files are correct:
- ✅ `frontend/src/pages/MultiDepotSimulation.tsx` exists
- ✅ `frontend/src/App.tsx` imports and routes it
- ✅ `frontend/src/components/layout/AppSidebar.tsx` has the link
- ✅ `frontend/src/services/api.ts` has `multiDepotSimulationApi`

