# Contributing Guide for Collaborators

Thank you for participating in this project! This document outlines code style, branching strategies, and testing requirements to keep our development fast and reliable.

## Development Setup

Follow these steps to establish a consistent local development environment:
1. Fork the repository and clone your fork locally.
2. Initialize virtual environments or install npm dependencies:
   - Python: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
   - Node: `npm install`
3. Install code quality verification libraries:
   - Python: `pip install pytest black flake8`
   - Node: `npm install --save-dev jest eslint prettier`
4. Run initial verification checks to ensure the existing test suite passes locally.

## Branch Strategy

We organize updates using clean branching structures:
- `feature/your-feature-name` for new components or updates.
- `fix/bug-fix-name` for patching issues.
- `docs/documentation-update` for file changes.
- `refactor/code-cleanup` for restructuring files without feature edits.

## Commit Conventions

We follow Conventional Commits. Commits must start with:
- `feat:` for new capabilities.
- `fix:` for code patches.
- `docs:` for documentation updates.
- `test:` for adding or updating tests.
- `refactor:` for code style changes with no feature updates.
- `chore:` for building setups or package updates.

Example:
`feat: add input preprocessing normalizer class`

## Pull Request Guidelines

1. Synchronize your local branch with upstream main before opening a PR.
2. Verify all linting checks and unit tests pass locally:
   - Python: `pytest` and `flake8`
   - Node: `npm run lint` and `npm test`
3. Document any API changes or configurations in the PR description.
4. Respond to feedback during code reviews.

## Code Style & Standards

- Write clean, readable code and follow standard conventions (PEP8 for Python, Prettier/ESLint for JS/TS).
- Keep functions modular, focused on a single responsibility.
- Write meaningful variable and function names.
- Document public APIs and write internal comments for complex logic.

## Review Process

- PRs require at least one maintainer review before merging.
- All unit tests must pass in the CI environment.
- Address outstanding review comments before code is approved.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)