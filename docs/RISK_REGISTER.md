# Risk Register

Things that could go wrong with this model and how I've tried to address them.

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R1 | Class imbalance biases toward majority class | High | High | SMOTE inside CV folds; track recall for minority class separately; optimize on PR-AUC in addition to F1 |
| R2 | Model overfits synthetic data patterns | Medium | High | Use held-out test set; compare complex models against LogReg baseline; noise in the data generator helps somewhat |
| R3 | Unfair performance across demographic subgroups | Medium | High | Run subgroup fairness check — F1 per group must be within 10% of population mean |
| R4 | Multiple hypothesis testing inflates Type I | High | Medium | Bonferroni and Benjamini-Hochberg correction applied to all 5 hypotheses |
| R5 | Cost assumptions are too optimistic | High | High | Tornado sensitivity analysis and Monte Carlo simulation to show how results change with different assumptions |
| R6 | Clinicians won't trust a black-box model | High | High | SHAP explanations + clinical narratives for top features; chose LogReg (interpretable) over XGBoost as final model |
| R7 | Synthetic data has impossible clinical values | Medium | Medium | Winsorization at clinically plausible bounds from medical reference ranges |
| R8 | Pipeline breaks on different systems | Medium | High | Docker support; pinned requirements.txt; config.py for all paths; global random seed |
| R9 | Dashboard shows too many technical metrics | High | Medium | Designed with "executive view" first (NMB, ROI); technical details in secondary tabs |
| R10 | Data leakage through premature scaling/encoding | Medium | High | All transformations inside sklearn Pipeline, fitted only on training folds |
