from app.domain.optimization.candidates import build_candidate_pool


def _valid_trainset(trainset_id: str = "T-001"):
    return {
        "trainset_id": trainset_id,
        "status": "STANDBY",
        "current_mileage": 1000,
        "max_mileage_before_maintenance": 50000,
        "fitness_certificates": {
            "rolling_stock": {"status": "VALID"},
            "signalling": {"status": "VALID"},
            "telecom": {"status": "VALID"},
        },
        "job_cards": {"open_cards": 0, "critical_cards": 0},
    }


def test_candidate_pool_separates_blocked_trainsets():
    blocked = _valid_trainset("T-002")
    blocked["job_cards"] = {"open_cards": 1, "critical_cards": 1}

    pool = build_candidate_pool([_valid_trainset(), blocked])

    assert [t["trainset_id"] for t in pool.eligible] == ["T-001"]
    assert [t["trainset_id"] for t in pool.blocked] == ["T-002"]
    assert pool.violations["T-002"]
    assert all(v.is_blocking for v in pool.violations["T-002"])


def test_candidate_pool_does_not_mutate_input():
    trainset = _valid_trainset()
    original_cards = dict(trainset["job_cards"])

    build_candidate_pool([trainset])

    assert trainset["job_cards"] == original_cards
