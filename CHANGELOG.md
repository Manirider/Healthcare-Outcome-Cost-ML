# Changelog

## v1.0.0 (2026-02-20)

- Full pipeline orchestration via `run_all.py`
- Added model evaluation with overfitting checks (train/test F1 gap)
- Integrated NMB cost-effectiveness into the Streamlit dashboard
- 43 unit tests passing
- Pinned dependencies in `requirements_pinned.txt`
- Docker support
- GitHub Actions CI

## v0.9.0 (2026-02-18)

- Refactored source code into modular `src/` structure
- Added Makefile for common commands
- Set up CI pipeline

## v0.1.0 (2026-01-15)

- Initial implementation: XGBoost and Logistic Regression models
- Baseline F1 ~0.85
- First Streamlit dashboard prototype
