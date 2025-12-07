import pytest
from app.services.fleet_planning import compute_required_trains, TimetableConfig, ServiceBand, LineParameters

def test_compute_required_trains_timetable():
    # Setup config
    config = TimetableConfig(
        service_bands=[
            ServiceBand(name="Peak", start_time="08:00", end_time="10:00", headway_min=10)
        ],
        line_params=LineParameters(line_runtime_min=40, turn_back_min=5),
        reserve_ratio=0.1
    )
    
    # Cycle time = 2*40 + 2*5 = 90 mins
    # Trains needed = ceil(90 / 10) = 9
    # Reserve = ceil(9 * 0.1) = 1
    # Total = 10
    
    result = compute_required_trains(timetable_config=config)
    
    assert result.required_service_trains == 9
    assert result.standby_buffer == 1
    assert result.total_required_trains == 10
    assert result.calculation_method == "timetable"

def test_compute_required_trains_override():
    result = compute_required_trains(override_count=20)
    
    assert result.required_service_trains == 20
    # Default reserve 0.15 -> ceil(20 * 0.15) = 3
    assert result.standby_buffer == 3
    assert result.total_required_trains == 23
    assert result.calculation_method == "override"

def test_compute_required_trains_legacy():
    # 24 hours, 12 hours/train -> 2 trains
    result = compute_required_trains(legacy_hours=24, avg_hours_per_train=12)
    
    assert result.required_service_trains == 2
    # Default reserve 0.15 -> ceil(2 * 0.15) = 1
    assert result.standby_buffer == 1
    assert result.total_required_trains == 3
    assert result.calculation_method == "legacy_hours"

def test_compute_required_trains_default():
    result = compute_required_trains()
    
    # Should use DEFAULT_TIMETABLE
    # Default: Runtime 45, Turnback 5 -> Cycle 100
    # Peak Headway 8 -> 100/8 = 12.5 -> 13 trains
    # Reserve 0.15 -> ceil(13 * 0.15) = 2
    # Total 15
    
    assert result.required_service_trains == 13
    assert result.standby_buffer == 2
    assert result.total_required_trains == 15
    assert result.calculation_method == "timetable_default"
