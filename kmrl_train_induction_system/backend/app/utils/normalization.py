# backend/app/utils/normalization.py
from typing import Any, Dict, List

def normalize_to_int(value: Any, default: int = 0) -> int:
    """Safely normalize value to integer, handling strings and edge cases"""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            # Strip whitespace and convert
            cleaned = value.strip()
            if not cleaned:
                return default
            return int(float(cleaned))  # Handle "2.0" strings
        except (ValueError, AttributeError):
            return default
    # For other types, try conversion
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def normalize_trainset_data(trainset: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize trainset data to ensure correct types."""
    # Normalize job_cards structure
    job_cards = trainset.get("job_cards", {})
    if not isinstance(job_cards, dict):
        job_cards = {}
    
    normalized_job_cards = {
        "open_cards": normalize_to_int(job_cards.get("open_cards"), 0),
        "critical_cards": normalize_to_int(job_cards.get("critical_cards"), 0),
        "job_cards_list": job_cards.get("job_cards_list", [])
    }
    
    # Create normalized copy
    normalized = trainset.copy()
    normalized["job_cards"] = normalized_job_cards
    
    # Normalize other critical fields
    if "current_mileage" in normalized:
        try:
            normalized["current_mileage"] = float(normalized["current_mileage"])
        except (ValueError, TypeError):
            normalized["current_mileage"] = 0.0
    
    if "max_mileage_before_maintenance" in normalized:
        try:
            max_mileage = normalized["max_mileage_before_maintenance"]
            if max_mileage in (None, "", 0):
                normalized["max_mileage_before_maintenance"] = float('inf')
            else:
                normalized["max_mileage_before_maintenance"] = float(max_mileage)
        except (ValueError, TypeError):
            normalized["max_mileage_before_maintenance"] = float('inf')
            
    return normalized
