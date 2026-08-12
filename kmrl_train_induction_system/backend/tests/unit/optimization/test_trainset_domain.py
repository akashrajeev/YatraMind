from app.domain.trainset import Trainset, TrainsetStatus


def test_trainset_domain_converts_current_storage_shape():
    trainset = Trainset.from_legacy_dict({
        "trainset_id": "T-101",
        "status": "standby",
        "current_mileage": "12000",
        "max_mileage_before_maintenance": "50000",
        "fitness_certificates": {"rolling_stock": {"status": "VALID"}},
        "job_cards": {"open_cards": "2", "critical_cards": "1"},
        "branding": {"current_advertiser": "Acme", "priority": "HIGH"},
        "requires_cleaning": True,
        "has_cleaning_slot": False,
    })

    assert trainset.trainset_id == "T-101"
    assert trainset.status is TrainsetStatus.STANDBY
    assert trainset.current_mileage == 12000.0
    assert trainset.job_cards.critical_cards == 1
    assert trainset.branding.advertiser == "Acme"
    assert trainset.cleaning.required is True


def test_trainset_domain_falls_back_for_unknown_status():
    trainset = Trainset.from_legacy_dict({"trainset_id": "T-102", "status": "BROKEN"})
    assert trainset.status is TrainsetStatus.STANDBY
