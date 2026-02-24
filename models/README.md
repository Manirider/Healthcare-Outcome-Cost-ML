# Models

Trained model artifacts.

## Files

- `best_model.pkl` — Logistic Regression (L2 regularized), the best performer (F1=0.92)
- `xgboost_model.pkl` — XGBoost gradient boosting
- `lightgbm_model.pkl` — LightGBM gradient boosting
- `mlp_model.pkl` — Multi-layer perceptron neural network
- `logistic_model.pkl` — same as best_model, saved under a consistent name
- `best_params.json` — hyperparameters used for training
- `train_indices.npy` / `test_indices.npy` — train/test split indices for reproducibility

## Notes

Logistic Regression outperformed the ensemble methods on this dataset, likely because the feature engineering step already linearized the key relationships. It also has better calibration and is more interpretable.
