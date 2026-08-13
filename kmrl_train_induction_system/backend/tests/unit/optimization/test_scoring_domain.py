from app.domain.optimization.scoring import OptimizationWeights, combined_score, tier2_score, tier3_score


def test_tier2_branding_and_defect_score():
    trainset = {
        "branding": {"current_advertiser": "Example", "priority": "HIGH"},
        "job_cards": {"open_cards": 2, "critical_cards": 0},
    }
    weights = OptimizationWeights()
    assert tier2_score(trainset, weights) == 200.0


def test_tier3_health_contribution_is_deterministic():
    trainset = {
        "current_mileage": 1000,
        "ml_health_score": 0.9,
        "requires_cleaning": False,
        "is_blocked": False,
    }
    first = tier3_score(trainset)
    second = tier3_score(trainset)
    assert first == second


def test_combined_score_tier2_dominates_tier3():
    trainset = {
        "branding": {"current_advertiser": "Example", "priority": "HIGH"},
        "job_cards": {"open_cards": 0, "critical_cards": 0},
        "ml_health_score": 0.1,
    }
    assert combined_score(trainset) > 3_000_000
