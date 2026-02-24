# Contributing

## Setup

```bash
git clone https://github.com/Manirider/Healthcare-Outcome-Cost-ML.git
cd Healthcare-Outcome-Cost-ML
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run tests before making changes:
```bash
pytest tests/
```

## Guidelines

- Add tests for new features or bug fixes
- Keep constants in `src/config.py` rather than hardcoding
- Run `python run_all.py` to make sure the full pipeline still works after your changes
- Use type hints for function signatures

## Submitting Changes

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Open a pull request
