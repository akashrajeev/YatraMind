# How to Access Multi-Depot Simulation

## Quick Access

1. **Start the frontend** (if not running):
   ```bash
   cd frontend
   npm run dev
   ```

2. **Open browser** and navigate to:
   ```
   http://localhost:5173/multi-depot-simulation
   ```
   OR
   - Click "Multi-Depot Simulation" in the sidebar (under Optimization)

3. **If you don't see it in sidebar**:
   - Make sure you're logged in as ADMIN or OPERATIONS_MANAGER
   - Refresh the page (Ctrl+F5 or Cmd+Shift+R)
   - Check browser console for errors

## Troubleshooting

### Page not loading?
1. **Check if dev server is running**:
   - Look for "VITE" output in terminal
   - Should show: `Local: http://localhost:5173`

2. **Clear browser cache**:
   - Press Ctrl+Shift+Delete (Windows) or Cmd+Shift+Delete (Mac)
   - Clear cached images and files
   - Refresh page

3. **Check browser console** (F12):
   - Look for any red errors
   - Check Network tab for failed API calls

4. **Verify route is registered**:
   - Open `frontend/src/App.tsx`
   - Look for route: `/multi-depot-simulation`

### Sidebar link not showing?
1. **Check user role**:
   - Only ADMIN and OPERATIONS_MANAGER can see this page
   - Logout and login as admin

2. **Check sidebar component**:
   - Open `frontend/src/components/layout/AppSidebar.tsx`
   - Look for "Multi-Depot Simulation" in navigationItems

### API errors?
1. **Start backend server**:
   ```bash
   cd backend
   python start_dev.py
   ```
   OR
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

2. **Check API endpoint**:
   - Backend should be running on `http://localhost:8000`
   - Test: `http://localhost:8000/api/v1/depots/presets`

3. **Check CORS**:
   - Backend should allow requests from `http://localhost:5173`

## Direct URL Access

If sidebar doesn't work, you can directly access:
```
http://localhost:5173/multi-depot-simulation
```

## What You Should See

1. **Simulation Control Panel**:
   - Fleet size buttons (25, 40, 60, 100)
   - Depot configuration cards
   - Add depot buttons
   - Run Simulation button

2. **After running simulation**:
   - Results tabs: Summary, Per-Depot, Transfers, Infrastructure
   - Global KPIs
   - Warnings (if any)
   - Transfer recommendations
   - Infrastructure suggestions

## Quick Test

1. Click "40 trains, 2 depots" stress test button
2. Click "Run Simulation"
3. Wait for results
4. Check "Global Summary" tab for KPIs

## Still Not Working?

1. **Restart dev server**:
   ```bash
   # Stop current server (Ctrl+C)
   cd frontend
   npm run dev
   ```

2. **Check for compilation errors**:
   ```bash
   cd frontend
   npm run typecheck
   ```

3. **Verify files exist**:
   - `frontend/src/pages/MultiDepotSimulation.tsx` should exist
   - `frontend/src/App.tsx` should import it
   - `frontend/src/components/layout/AppSidebar.tsx` should have the link

4. **Check browser console** for specific error messages

