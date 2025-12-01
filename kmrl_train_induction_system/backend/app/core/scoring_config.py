# backend/app/core/scoring_config.py

# Centralized scoring weights for Train Induction Optimization
# Used by both the Optimizer service and Explainability utils to ensure consistency.

SCORING_WEIGHTS = {
    # Tier 2: High Priority Soft Objectives
    "BRANDING_OBLIGATION": 300.0,          # Increased from 250.0 to outweigh 5 minor defects (5 * -50 = -250)
    "MINOR_DEFECT_PENALTY_PER_DEFECT": -50.0,
    
    # Tier 3: Optimization Soft Objectives
    "MILEAGE_BALANCING": 50.0,
    "CLEANING_DUE_PENALTY": -30.0,
    "SHUNTING_COMPLEXITY_PENALTY": -20.0
}
