# Contributing to Scorebook

Thank you for your interest in contributing to Scorebook! This document provides guidelines and instructions for setting up your development environment using Poetry.

## Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation) installed on your system

## Setting Up Your Development Environment

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd scorebook
   ```

2. **Install Dependencies**

   Install all project dependencies using Poetry:

   ```bash
   poetry install
   ```

   This will install all dependencies specified in `pyproject.toml`, including development dependencies.

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and run tests. To set up pre-commit hooks, run:

```bash
poetry run pre-commit install
```

This will install the pre-commit hooks, which will run automatically on every git commit.

## Testing & Code Quality

Pre-commit hooks automatically run both tests and code quality checks including:
- **pytest**: Unit tests
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **autoflake**: Remove unused imports

To run all checks manually:

```bash
poetry run pre-commit run --all-files
```

To run only tests:

```bash
poetry run pytest
```

Look into `.pre-commit-config.yaml` to check how each tool is configured.

## Submitting Changes

We follow [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow), a lightweight, branch-based workflow that supports teams and projects where deployments are made regularly. The main branch should always be deployable, and new features are developed in feature branches.

Before submitting any changes, please ensure:
1. Pre-commit hooks are installed (`poetry run pre-commit install`)
2. All pre-commit checks pass (`poetry run pre-commit run --all-files`)

When you create a pull request, GitHub Actions will automatically run the same checks in the pre-commit hooks on the codebase. Ensuring that pre-commit hooks pass before submitting a pull request will ensure smoother and faster code reviews. Pull reviews that fail automatic testing will not be reviewed until the issues have been resolved.

### For External Contributors
1. Fork the repository to your GitHub account
2. Clone your fork locally
3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/marco/scorebook.git
   ```
4. Create a new branch for your changes
5. Make your changes and commit them
6. Push your branch to your fork
7. Create a pull request from your fork to the main repository

### For Direct Contributors
1. Create a new branch for your changes
2. Make your changes and commit them
3. Push your branch to the repository
4. Create a pull request

**Important**: Pull requests will only be merged if all pre-commit checks pass. Make sure to run `poetry run pre-commit run --all-files` locally before submitting your changes.

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write your code
   - Add tests for new functionality
   - Update documentation if needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

   **Note**: Make sure you have installed pre-commit hooks (`poetry run pre-commit install`) before committing. The hooks will run automatically and prevent the commit if any checks fail. Remember that pull requests with failing tests will not be reviewed until issues are resolved.

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

Thank you for contributing to Scorebook!
