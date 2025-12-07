# Train Count Determination Method

## Current Method: **Timetable-Driven Calculation**

The system now uses a **timetable-driven approach** as the primary method for determining the number of trains required for service.

---

## How It Works

### 1. **Timetable-Driven Calculation** (Primary Method)

The system calculates required trains based on:
- **Service Bands**: Peak and off-peak periods with different headways
- **Line Parameters**: Runtime and turnback time
- **Cycle Time**: Round trip time for a train
- **Headway**: Time between consecutive trains

**Formula:**
```
Cycle Time = 2 × Line Runtime + 2 × Turnback Time
Trains Needed per Band = ceil(Cycle Time / Headway)
Required Service Trains = max(Trains Needed across all bands)
Standby Buffer = ceil(Required Service Trains × Reserve Ratio)
Total Required = Required Service Trains + Standby Buffer
```

**Default Configuration:**
- **Line Runtime**: 45 minutes (Aluva ↔ Petta)
- **Turnback Time**: 5 minutes
- **Cycle Time**: 100 minutes (2×45 + 2×5)
- **Service Bands**:
  - Morning Peak (08:00-11:00): Headway = 8 min → **13 trains**
  - Evening Peak (17:00-20:00): Headway = 8 min → **13 trains**
  - Off-Peak (06:00-22:00): Headway = 15 min → **7 trains**
- **Reserve Ratio**: 15%

**Result**: `required_service_trains = 13` (from peak bands), `standby_buffer = 2`, `total = 15`

### 2. **Manual Override** (Optional)

If `required_service_count` is provided in the API request, it overrides the timetable calculation:
- Uses the provided count directly as `required_service_trains`
- Still calculates standby buffer (15% of service trains)
- Useful for special operations or testing

---

## Priority Order

1. **Manual Override** (`required_service_count`) - if provided
2. **Timetable-Driven** (default) - uses `DEFAULT_TIMETABLE` configuration
3. **Service Date** (`service_date`) - can be used for future timetable lookups (currently uses default)

---

## Implementation

**File**: `backend/app/services/fleet_planning.py`

**Function**: `compute_required_trains()`

**Usage in Optimizer**:
```python
fleet_req = compute_required_trains(
    service_date=request.service_date,
    timetable_config=None,  # Uses default
    override_count=request.required_service_count
)

required_service_trains = fleet_req.required_service_trains
target_service = min(required_service_trains, eligible_count)
```

---

## Removed Legacy Methods

### ❌ **Removed: `required_service_hours`**

The old method that treated service hours as train count has been **completely removed**:
- ❌ No longer in `OptimizationRequest` model
- ❌ No longer in `compute_required_trains()` function
- ❌ No longer in API endpoints
- ❌ No longer in optimizer code
- ❌ No longer in what-if simulator

**Why Removed:**
- Was error-prone (treated hours as train count)
- Didn't account for actual operational requirements
- Timetable-driven method is more accurate and scalable

---

## API Usage

### Current API Request Format:

```json
{
  "target_date": "2025-01-15T00:00:00Z",
  "service_date": "2025-01-15",  // Optional: for timetable lookup
  "required_service_count": 14    // Optional: manual override
}
```

**If `required_service_count` is provided**: Uses that count directly  
**If not provided**: Uses timetable-driven calculation with default configuration

---

## Benefits of Timetable-Driven Method

1. **Accurate**: Based on actual operational requirements (headway, cycle time)
2. **Scalable**: Automatically adjusts for different service patterns
3. **Configurable**: Service bands and line parameters can be updated
4. **Transparent**: Clear calculation method with explainable results
5. **Future-Ready**: Supports date-based timetable lookups for different days

---

## Configuration

Default timetable configuration is in `fleet_planning.py`:
- Can be moved to database/config file for dynamic updates
- Supports different service bands for different days
- Can be customized per depot or line

---

**Last Updated**: 2025-01  
**Status**: ✅ Active - Timetable-driven method is the primary and only method

