"""Unit tests for bay number parsing with multi-format support"""
import pytest
from app.services.stabling_optimizer import StablingGeometryOptimizer


@pytest.fixture
def optimizer():
    return StablingGeometryOptimizer()


def test_parse_bay_BAY_format(optimizer):
    """Test _BAY_ format parsing"""
    assert optimizer._extract_bay_number("Aluva_BAY_05") == 5
    assert optimizer._extract_bay_number("Petta_BAY_12") == 12
    assert optimizer._extract_bay_number("_BAY_5") == 5


def test_parse_bay_Bay_space_format(optimizer):
    """Test 'Bay 5' format parsing"""
    assert optimizer._extract_bay_number("Bay 5") == 5
    assert optimizer._extract_bay_number("Bay 05") == 5
    assert optimizer._extract_bay_number("bay 12") == 12


def test_parse_bay_B_dash_format(optimizer):
    """Test 'B-5' format parsing"""
    assert optimizer._extract_bay_number("B-5") == 5
    assert optimizer._extract_bay_number("B-05") == 5
    assert optimizer._extract_bay_number("b-12") == 12


def test_parse_bay_B_no_dash_format(optimizer):
    """Test 'B5' format parsing"""
    assert optimizer._extract_bay_number("B5") == 5
    assert optimizer._extract_bay_number("B05") == 5
    assert optimizer._extract_bay_number("b12") == 12


def test_parse_bay_bay_underscore_format(optimizer):
    """Test 'bay_5' format parsing"""
    assert optimizer._extract_bay_number("bay_5") == 5
    assert optimizer._extract_bay_number("bay_05") == 5
    assert optimizer._extract_bay_number("BAY_12") == 12


def test_parse_bay_numeric_only(optimizer):
    """Test numeric-only format parsing"""
    assert optimizer._extract_bay_number("5") == 5
    assert optimizer._extract_bay_number("05") == 5
    assert optimizer._extract_bay_number("12") == 12


def test_parse_bay_unparseable_returns_none(optimizer):
    """Test that unparseable formats return None"""
    assert optimizer._extract_bay_number("") is None
    assert optimizer._extract_bay_number("invalid") is None
    assert optimizer._extract_bay_number("Bay X") is None
    assert optimizer._extract_bay_number("No bay") is None
    assert optimizer._extract_bay_number(None) is None


def test_parse_bay_case_insensitive(optimizer):
    """Test that parsing is case-insensitive"""
    assert optimizer._extract_bay_number("BAY_5") == 5
    assert optimizer._extract_bay_number("bay_5") == 5
    assert optimizer._extract_bay_number("Bay 5") == 5
    assert optimizer._extract_bay_number("b-5") == 5
    assert optimizer._extract_bay_number("B-5") == 5








