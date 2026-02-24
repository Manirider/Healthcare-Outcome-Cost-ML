# Model Card: Healthcare Outcome Predictor

**Model Type:** Logistic Regression (L2-regularized)  
**Training Data:** 8,000 synthetic patient records  
**License:** MIT

## Intended Use

Risk stratification for hospitalized patients — specifically, identifying who should receive enhanced care protocols vs. standard treatment. Intended users are clinical administrators and triage staff.

This is NOT a diagnostic tool. It predicts treatment outcomes based on existing clinical measurements; it does not diagnose conditions.

## Performance

| Metric | Value |
|--------|-------|
| F1 Score | 0.92 |
| AUC-ROC | 0.99 |
| Calibration (ECE) | 0.018 |
| Brier Score | 0.042 |

## Key Features

Top predictors (by SHAP importance):
- CVD Risk Score (composite)
- Age
- BMI
- Treatment Intensity Ratio
- HbA1c

## Fairness

Subgroup performance audited across age, sex, and BMI categories. Maximum F1 disparity across any subgroup is less than 10%.

## Limitations

- Trained on synthetic data — must be retrained on real hospital data before deployment
- Does not account for rare comorbidities (e.g., autoimmune conditions)
- Cost-effectiveness projections depend heavily on assumed cost parameters
- No temporal validation since the data has no time dimension

## Loading the Model

```python
import joblib
model = joblib.load('models/best_model.pkl')
```
