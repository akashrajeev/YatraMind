"""Test that results is always coerced to array"""
import pytest
import json
from pathlib import Path
from app.api.simulation import _ensure_results_is_array


def test_ensure_results_is_array_with_list():
    """Test that list results remain as list"""
    data = {"results": [{"type": "baseline"}, {"type": "scenario"}]}
    result = _ensure_results_is_array(data)
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2


def test_ensure_results_is_array_with_object():
    """Test that object results are converted to array"""
    data = {"results": {"type": "baseline", "kpis": {}}}
    result = _ensure_results_is_array(data)
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1
    assert result["results"][0]["type"] == "baseline"


def test_ensure_results_is_array_with_none():
    """Test that None results become empty array"""
    data = {"results": None}
    result = _ensure_results_is_array(data)
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 0


def test_ensure_results_is_array_with_empty_dict():
    """Test that empty dict results become empty array"""
    data = {"results": {}}
    result = _ensure_results_is_array(data)
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 0


def test_ensure_results_is_array_missing_key():
    """Test that missing results key gets empty array"""
    data = {"other_field": "value"}
    result = _ensure_results_is_array(data)
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 0


def test_load_saved_simulation_with_object_results():
    """Test loading a saved simulation that has object instead of array"""
    # Simulate loading from file with object results
    data_from_file = {
        "simulation_id": "test-123",
        "results": {"type": "baseline", "kpis": {}}  # Wrong format
    }
    
    # Apply coercion
    fixed_data = _ensure_results_is_array(data_from_file)
    
    # Should now be array
    assert isinstance(fixed_data["results"], list)
    assert len(fixed_data["results"]) == 1







