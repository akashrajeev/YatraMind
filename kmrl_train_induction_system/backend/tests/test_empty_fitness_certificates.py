"""Unit tests for empty fitness certificates handling"""
import pytest
from app.services.optimizer import TrainInductionOptimizer


@pytest.fixture
def optimizer():
    return TrainInductionOptimizer()


def test_empty_fitness_certificates_dict_is_critical_failure(optimizer):
    """Test that empty dict {} is treated as critical failure"""
    trainset = {
        "trainset_id": "T-001",
        "fitness_certificates": {},  # Empty dict
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == True


def test_missing_fitness_certificates_is_critical_failure(optimizer):
    """Test that missing fitness_certificates key is critical failure"""
    trainset = {
        "trainset_id": "T-001",
        # No fitness_certificates key
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == True


def test_invalid_fitness_certificates_type_is_critical_failure(optimizer):
    """Test that non-dict fitness_certificates is critical failure"""
    trainset = {
        "trainset_id": "T-001",
        "fitness_certificates": "invalid",  # Not a dict
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == True


def test_missing_required_certificates_is_critical_failure(optimizer):
    """Test that missing required certificates (rolling_stock, signalling, telecom) is failure"""
    trainset = {
        "trainset_id": "T-001",
        "fitness_certificates": {
            "rolling_stock": {"status": "VALID"},  # Missing signalling and telecom
        },
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == True


def test_valid_fitness_certificates_passes(optimizer):
    """Test that valid certificates pass the check"""
    trainset = {
        "trainset_id": "T-001",
        "fitness_certificates": {
            "rolling_stock": {"status": "VALID"},
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"}
        },
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == False


def test_expired_certificate_is_critical_failure(optimizer):
    """Test that expired certificate is critical failure"""
    trainset = {
        "trainset_id": "T-001",
        "fitness_certificates": {
            "rolling_stock": {"status": "EXPIRED"},  # Expired
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"}
        },
        "job_cards": {"open_cards": 0, "critical_cards": 0}
    }
    
    assert optimizer._has_critical_failure(trainset) == True








