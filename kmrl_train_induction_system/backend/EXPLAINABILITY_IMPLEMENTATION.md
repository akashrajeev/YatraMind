# Explainability & Sustainability Implementation

## Overview

This implementation provides comprehensive explainability and sustainability features for the KMRL Train Induction System. Each final assignment entry now returns detailed explanations including composite scores, contributing factors, risks, violations, and SHAP values.

## Features Implemented

### 1. Enhanced Data Models

**File**: `app/models/trainset.py`

- Added `ShapFeature` model for feature attribution data
- Enhanced `InductionDecision` model with explainability fields:
  - `score`: Composite score for the assignment
  - `top_reasons`: Top 3 contributing positive reasons
  - `top_risks`: Top 3 negative reasons
  - `violations`: List of rule violations
  - `shap_values`: Top 5 features and their impact

### 2. Comprehensive Explainability Engine

**File**: `app/utils/explainability.py`

#### Core Functions:

- `calculate_composite_score()`: Calculates weighted composite score based on:
  - Fitness score (30%)
  - Reliability score (25%)
  - Branding priority (20%)
  - Mileage balance (15%)
  - Operational efficiency (10%)

- `generate_comprehensive_explanation()`: Generates complete explanation including:
  - Composite score
  - Top reasons and risks
  - Rule violations
  - SHAP feature attributions

- `generate_shap_values()`: Simulates SHAP values for top features:
  - Sensor Health Score
  - Predicted Failure Risk
  - Mileage Usage Ratio
  - Branding Priority
  - Fitness Certificate Score
  - Maintenance Priority

- `detect_rule_violations()`: Identifies rule violations:
  - Certificate expiry violations
  - Critical job card violations
  - Mileage limit violations
  - Maintenance status violations
  - Cleaning slot violations

#### Rendering Functions:

- `render_explanation_html()`: Generates printable HTML explanations
- `render_explanation_text()`: Generates plain text explanations

### 3. Enhanced Jinja Templates

**File**: `app/templates/explanation.j2`

- Professional HTML template with CSS styling
- Responsive design for both screen and print
- Color-coded elements:
  - Green for positive reasons
  - Red for risks and violations
  - Visual indicators for SHAP feature impacts (↑↓→)
- Print-friendly styling

### 4. Updated Optimization Engine

**File**: `app/services/optimizer.py`

- Integrated explainability into optimization decisions
- All decision types (INDUCT, STANDBY, MAINTENANCE) now include comprehensive explanations
- Both OR-Tools and fallback optimization methods enhanced

### 5. New API Endpoints

**File**: `app/api/optimization.py`

#### Individual Explanation Endpoint:
```
GET /optimization/explain/{trainset_id}?decision=INDUCT&format=json
```

**Parameters:**
- `trainset_id`: ID of the trainset
- `decision`: Assignment decision (INDUCT/STANDBY/MAINTENANCE)
- `format`: Output format (json/html/text)

#### Batch Explanation Endpoint:
```
POST /optimization/explain/batch
```

**Request Body:**
```json
[
  {"trainset_id": "TS001", "decision": "INDUCT"},
  {"trainset_id": "TS002", "decision": "STANDBY"}
]
```

**Parameters:**
- `format`: Output format (json/html/text)

## Example Output

### JSON Response:
```json
{
  "score": 0.847,
  "top_reasons": [
    "All department certificates valid",
    "Low predicted failure probability",
    "Available cleaning slot before dawn"
  ],
  "top_risks": [
    "SAFETY certificate expiring soon"
  ],
  "violations": [],
  "shap_values": [
    {
      "name": "Sensor Health Score",
      "value": 0.85,
      "impact": "positive"
    },
    {
      "name": "Predicted Failure Risk",
      "value": 0.15,
      "impact": "positive"
    }
  ],
  "trainset_id": "TS001",
  "role": "INDUCT",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### HTML Output:
Professional HTML with:
- Styled explanation cards
- Color-coded reasons and risks
- Feature impact visualizations
- Print-friendly formatting

### Text Output:
```
Trainset TS001 - Decision: INDUCT
Composite Score: 0.847

Top Reasons:
  • All department certificates valid
  • Low predicted failure probability
  • Available cleaning slot before dawn

Top Risks:
  • SAFETY certificate expiring soon

Feature Attributions:
  • Sensor Health Score: 0.8500 ↑
  • Predicted Failure Risk: 0.1500 ↑
```

## Scoring Algorithm

The composite score is calculated using weighted factors:

1. **Fitness Score (30%)**: Based on certificate validity and job card status
2. **Reliability Score (25%)**: Based on sensor health and predicted failure risk
3. **Branding Priority (20%)**: Based on advertising contract priority
4. **Mileage Balance (15%)**: Prefers balanced usage (not too low/high)
5. **Operational Efficiency (10%)**: Based on cleaning slots and maintenance compliance

Decision multipliers:
- INDUCT: 1.0x (full score)
- STANDBY: 0.7x (reduced score)
- MAINTENANCE: 0.3x (lowest score)

## Rule Violations Detection

The system detects various rule violations:

- **Certificate Violations**: Expired or expiring certificates
- **Job Card Violations**: Critical job cards pending
- **Mileage Violations**: Exceeding maintenance mileage limits
- **Status Violations**: Attempting to induct trainsets in maintenance
- **Cleaning Violations**: Missing required cleaning slots

## SHAP Values Simulation

The system simulates SHAP values for key features:

- **Sensor Health Score**: Higher values indicate better reliability
- **Predicted Failure Risk**: Lower values indicate better reliability
- **Mileage Usage Ratio**: Moderate usage preferred
- **Branding Priority**: Higher priority increases score
- **Fitness Certificate Score**: Based on certificate validity
- **Maintenance Priority**: Based on job cards and mileage

## Usage Examples

### Generate Explanation for Single Trainset:
```python
from app.utils.explainability import generate_comprehensive_explanation

explanation = generate_comprehensive_explanation(trainset_data, "INDUCT")
print(f"Score: {explanation['score']:.3f}")
print(f"Reasons: {explanation['top_reasons']}")
```

### Render HTML Explanation:
```python
from app.utils.explainability import render_explanation_html

html_content = render_explanation_html({
    "trainset_id": "TS001",
    "role": "INDUCT",
    **explanation
})
```

### API Usage:
```bash
# Get explanation for specific trainset
curl -H "X-API-Key: your-key" \
  "http://localhost:8000/optimization/explain/TS001?decision=INDUCT&format=html"

# Batch explanations
curl -X POST -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '[{"trainset_id": "TS001", "decision": "INDUCT"}]' \
  "http://localhost:8000/optimization/explain/batch?format=json"
```

## Dependencies Added

- `jinja2==3.1.2`: Template rendering
- `shap==0.42.1`: SHAP value calculations (for production ML models)

## Testing

The implementation includes comprehensive test coverage:

- Composite score calculations
- Explanation generation
- HTML and text rendering
- Different scenarios (high risk, perfect trainsets)
- Edge cases and error handling

## Future Enhancements

1. **Real SHAP Integration**: Replace simulation with actual SHAP explainer
2. **Advanced ML Models**: Integrate with production ML models
3. **Customizable Weights**: Allow dynamic weight adjustment
4. **Historical Analysis**: Track explanation trends over time
5. **Performance Metrics**: Add explanation quality metrics

## Conclusion

This implementation provides comprehensive explainability and sustainability features that enable:

- **Transparent Decision Making**: Clear explanations for all assignment decisions
- **Risk Assessment**: Identification of potential issues and violations
- **Feature Attribution**: Understanding of which factors influence decisions
- **Printable Reports**: Professional HTML and text outputs for documentation
- **API Integration**: Easy integration with frontend applications

The system maintains backward compatibility while adding powerful new explainability capabilities that enhance operational transparency and decision support.
