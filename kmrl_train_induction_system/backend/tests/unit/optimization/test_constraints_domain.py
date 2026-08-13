from app.domain.optimization.constraints import REQUIRED_FITNESS_CERTIFICATES, validate_trainset_safety
from app.domain.optimization.types import ConstraintCode, Severity


def valid_trainset():
    return {
        "trainset_id": "T-001",
        "status": "STANDBY",
        "current_mileage": 10000,
        "max_mileage_before_maintenance": 50000,
        "fitness_certificates": {
            name: {"status": "VALID"} for name in REQUIRED_FITNESS_CERTIFICATES
        },
        "job_cards": {"critical_cards": 0, "open_cards": 0},
        "requires_cleaning": False,
        "has_cleaning_slot": True,
    }


def test_valid_trainset_has_no_blocking_constraints():
    violations = validate_trainset_safety(valid_trainset())
    assert violations == []


def test_missing_certificates_is_critical():
    trainset = valid_trainset()
    trainset["fitness_certificates"] = {}

    violations = validate_trainset_safety(trainset)

    assert any(v.code == ConstraintCode.MISSING_FITNESS_CERTIFICATES for v in violations)
    assert all(v.severity == Severity.CRITICAL for v in violations)


def test_expired_certificate_is_structured_not_textual():
    trainset = valid_trainset()
    trainset["fitness_certificates"]["telecom"] = {"status": "EXPIRED"}

    violations = validate_trainset_safety(trainset)

    assert any(v.code == ConstraintCode.EXPIRED_FITNESS_CERTIFICATE for v in violations)
    assert any(v.is_blocking for v in violations)


def test_critical_job_cards_are_blocking():
    trainset = valid_trainset()
    trainset["job_cards"]["critical_cards"] = "2"

    violations = validate_trainset_safety(trainset)

    assert any(v.code == ConstraintCode.CRITICAL_JOB_CARD for v in violations)
