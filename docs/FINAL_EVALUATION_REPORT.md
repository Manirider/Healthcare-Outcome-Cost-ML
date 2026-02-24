# Evaluation Report

## Overview

This document summarizes the evaluation of the Healthcare Outcome Prediction system against the project requirements. It covers each deliverable and notes any limitations or areas for future work.

## Data Preparation

The synthetic dataset contains 8,000 patient records across 17 features. Missing values were introduced following a Missing At Random (MAR) mechanism to simulate real clinical data collection. MICE imputation was chosen over simpler methods (mean/median) because it preserves the correlation structure between clinical variables — important when features like SBP and DBP are biologically linked.

Outlier handling uses clinical Winsorization rather than IQR-based deletion. The reasoning: a BMI of 45 is unusual but medically real. Deleting these patients would introduce selection bias. Instead, values are capped at clinically plausible upper/lower bounds sourced from medical reference ranges.

## Statistical Analysis

Five hypotheses were tested. Each test includes:
- Assumption checking (Shapiro-Wilk for normality, Levene's for variance homogeneity)
- Appropriate test selection based on assumption results
- Effect size computation
- Power analysis
- Multiple testing correction (Bonferroni and Benjamini-Hochberg)

All five hypotheses were statistically significant after correction. The largest effect sizes were observed for CVD risk (Cohen's d = 2.12) and metabolic syndrome score (Cramer's V = 0.43).

## Feature Engineering

Eight features were created, each with documented clinical rationale. The ablation study confirms that each feature contributes positively to model performance. The most impactful engineered feature is the CVD risk composite score, which draws on established Framingham risk factor methodology.

## Model Performance

| Model | F1 | AUC-ROC | Brier | ECE |
|-------|-----|---------|-------|-----|
| Logistic Regression | 0.9206 | 0.9876 | 0.0421 | 0.018 |
| XGBoost | 0.9013 | 0.9855 | 0.0465 | 0.023 |
| LightGBM | 0.8988 | 0.9855 | 0.0467 | 0.025 |
| MLP | 0.8936 | 0.9830 | 0.0518 | 0.031 |

All models exceed the F1 >= 0.80 target. Logistic Regression performs best — likely because the feature engineering linearized the key relationships.

Subgroup fairness analysis shows no demographic group has F1 deviation greater than 10% from the population mean.

## Cost-Effectiveness

ICER and NMB calculations confirm that the Enhanced treatment arm is cost-effective at standard willingness-to-pay thresholds ($50k-$100k/QALY). Sensitivity analysis (tornado plot) shows the result is most sensitive to the cost of bad outcomes and QALY assumptions.

The projected savings from model-guided treatment allocation are approximately $700k/year per 1,000 patients.

## Limitations

- The dataset is synthetic, so model performance may not transfer to real clinical data
- Cost assumptions are illustrative and would need calibration against actual hospital billing data
- The QALY estimates are simplified; real pharmacoeconomic analysis would use more granular utility weights
- SHAP analysis is computed on the logistic regression model; tree-based models would show different importance patterns
- No temporal validation (time-series split) since the data lacks a time dimension

## Deliverables Checklist

- [x] Synthetic dataset with realistic clinical distributions
- [x] EDA with 24 visualizations
- [x] 5 hypothesis tests with correction for multiple comparisons
- [x] 8 engineered features with clinical rationale
- [x] 4 models trained and compared (all F1 > 0.80)
- [x] SHAP interpretability analysis
- [x] Cost-effectiveness analysis (ICER, NMB, sensitivity)
- [x] Interactive Streamlit dashboard
- [x] Executive summary PDF report
- [x] 43 unit tests
- [x] Docker support
- [x] Reproducible pipeline (run_all.py)
