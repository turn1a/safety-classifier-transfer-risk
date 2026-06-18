# transfer-risk task runner — https://just.systems
# Install: `brew install just`. List recipes: `just --list`.

# Load .env (telemetry opt-out, optional HF token, Apple-Silicon/torch runtime flags)
# into every recipe's environment. See .env.example.
set dotenv-load := true

# AWS profile and region for the cloud sweep recipes (see infra/ and SPEC.md).
aws_profile := "transfer-risk-cli"
aws_region := "eu-central-1"

default: check

# Install dependencies and the pre-commit / pre-push hooks.
install:
    uv sync
    uv run pre-commit install --install-hooks

# Download the offline NLP assets the attacks need (NLTK data, counter-fitted
# embeddings, the sentence-transformers encoder). Run once before a full attack run.
setup-data:
    uv run python -m transfer_risk.scripts.fetch_assets

# Auto-fix formatting (ruff + mdformat).
fmt:
    uv run ruff format .
    uv run mdformat .

# Lint (ruff + docstring coverage). Fails on any diagnostic — never ignore.
lint:
    uv run ruff check .
    uv run interrogate src

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

# Fast end-to-end slice. KEDRO_ENV=thin is required so the dynamic pipelines read the reduced
# structure (3 surrogates, 3 recipes) at build time, not just at run time. e.g. `just run-thin`.
run-thin *args:
    KEDRO_ENV=thin uv run kedro run {{args}}

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

# --- cloud sweep (one fat spot box; see infra/ and the plan) ---------------
# Typical flow: cloud-stage -> cloud-up -> cloud-finish. The catalog owns all S3 I/O via
# ${globals:...}; there is no aws s3 sync of project data. cloud-stage trains the pool once with
# --env cloud (writing splits/surrogates/ONNX to S3 and the rest locally); the box runs only the
# attacks; cloud-finish runs the downstream (transfer/risk/reporting + MLflow) locally.

# Create the S3 bucket, then build + stage everything the box reads: run data, models, similarity
# under --env cloud so the boundary artifacts (splits, surrogate checkpoints, ONNX) land on S3 and
# the rest stay local. One training pass; re-run is a no-op with --only-missing-outputs.
cloud-stage:
    #!/usr/bin/env bash
    set -euo pipefail
    terraform -chdir=infra init -input=false
    terraform -chdir=infra apply -auto-approve -target=aws_s3_bucket.exchange
    bucket="$(terraform -chdir=infra output -raw bucket_name)"
    AWS_PROFILE={{aws_profile}} AWS_REGION={{aws_region}} TR_BUCKET="$bucket" TR_REGION={{aws_region}} \
      uv run --group cloud kedro run --env cloud --pipeline "data,models,similarity" --only-missing-outputs

# Provision the rest of the infra (IAM/SG/spot box) and start the sweep. The box clones the repo
# at terraform.tfvars repo_ref, runs the attacks under --env cloud (ParallelRunner,
# --only-missing-outputs), reading and writing S3 through the catalog, then self-terminates.
cloud-up:
    terraform -chdir=infra init -input=false
    terraform -chdir=infra apply -auto-approve
    @echo "watch: just cloud-logs  |  shell: just cloud-attach  |  results: just cloud-finish"

# Run the downstream locally against the S3 partitions: transfer/risk/reporting under --env cloud
# read the per-cell adversarial partitions from S3 and write results, figures, and MLflow locally.
cloud-finish:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terraform -chdir=infra output -raw bucket_name)"
    AWS_PROFILE={{aws_profile}} AWS_REGION={{aws_region}} TR_BUCKET="$bucket" TR_REGION={{aws_region}} \
      uv run --group cloud kedro run --env cloud --pipeline "transfer,risk,reporting"

# Stream the box's bootstrap/run log from S3.
cloud-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terraform -chdir=infra output -raw bucket_name)"
    aws s3 cp "s3://$bucket/logs/latest/run.log" - --profile {{aws_profile}} --region {{aws_region}}

# Open a shell on the box via SSM Session Manager (no SSH); then `tmux attach -t run`.
cloud-attach:
    #!/usr/bin/env bash
    set -euo pipefail
    iid="$(terraform -chdir=infra output -raw instance_id)"
    aws ssm start-session --target "$iid" --profile {{aws_profile}} --region {{aws_region}}

# Tear down everything (box, IAM, SG, bucket). Run cloud-finish first: the bucket holds the
# adversarial partitions, and the final results are produced locally by the downstream run, so
# they are safe once cloud-finish has completed.
cloud-down:
    terraform -chdir=infra destroy -auto-approve
