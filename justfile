# transfer-risk task runner — https://just.systems
# Install: `brew install just`. List recipes: `just --list`.

# Opt out of kedro-telemetry for every recipe.
export DO_NOT_TRACK := "1"

default: check

# Install dependencies and the pre-commit / pre-push hooks.
install:
    uv sync
    uv run pre-commit install --install-hooks

# Auto-fix formatting (ruff + mdformat).
fmt:
    uv run ruff format .
    uv run mdformat .

# Lint (ruff). Fails on any diagnostic — never ignore.
lint:
    uv run ruff check .

# Type-check src/ (mypy strict).
type:
    uv run mypy src

# Run the test suite with coverage (gate enforced via pyproject.toml).
test *args:
    uv run pytest {{args}}

# All quality gates in one shot.
check: lint type test

# Run every pre-commit and pre-push hook against all files.
hooks:
    uv run pre-commit run --all-files --hook-stage pre-commit
    uv run pre-commit run --all-files --hook-stage pre-push

# Run a Kedro pipeline (default: the full chain). e.g. `just run --pipeline smoke`.
run *args:
    uv run kedro run {{args}}

# Open the interactive pipeline DAG in the browser.
viz *args:
    uv run kedro viz run {{args}}

# Export the DAG as a static site for embedding in the docs/blog.
viz-build:
    uv run kedro viz build

# Render the Quarto site to docs/_site (needs the `quarto` CLI on PATH).
docs:
    quarto render docs

# Live-preview the Quarto site.
docs-preview:
    quarto preview docs

# Launch the local MLflow UI against the project tracking store.
mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
