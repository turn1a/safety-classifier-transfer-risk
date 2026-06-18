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

# Export each transformer surrogate to ONNX for `use_onnx` attack runs (onnxruntime is ~2-3x
# faster per query on CPU). Runs optimum in a throwaway `uvx` env — optimum conflicts with
# transformers 5 — so it never touches the main environment. Run once before an ONNX sweep.
export-onnx:
    uv run python -m transfer_risk.scripts.export_onnx

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
# Typical flow: export-onnx -> cloud-push -> cloud-up -> cloud-pull -> run --from-nodes evaluate_transfer.
# The downstream (transfer/risk/reporting + MLflow) runs locally, so MLflow stays a single
# local sqlite writer and results land here.

# Provision the spot box and start the sweep (creates the bucket/IAM/SG on first run). The box
# clones the repo at terraform.tfvars repo_ref, runs the attacks under --env cloud, self-terminates.
cloud-up:
    terraform -chdir=infra init -input=false
    terraform -chdir=infra apply -auto-approve
    @echo "watch: just cloud-logs  |  shell: just cloud-attach  |  results: just cloud-pull"

# Upload the cached inputs the box needs: canonical data, splits, and surrogates (incl. the
# exported ONNX). Run after `just export-onnx`, and again whenever surrogates change.
cloud-push:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terraform -chdir=infra output -raw bucket_name)"
    for layer in 03_primary 05_model_input 06_models; do
      aws s3 sync "data/$layer/" "s3://$bucket/data/$layer/" --profile {{aws_profile}} --region {{aws_region}}
    done

# Pull the adversarial partitions the box produced back into the local data/ dir.
cloud-pull:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terraform -chdir=infra output -raw bucket_name)"
    aws s3 sync "s3://$bucket/data/07_model_output/adversarial_examples/" data/07_model_output/adversarial_examples/ --profile {{aws_profile}} --region {{aws_region}}

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

# Tear down everything (box, IAM, SG, bucket). Pull first: the bucket holds the intermediate
# partitions, while the final results are produced locally by the downstream run, so they are
# safe once you have run `just cloud-pull` and the local downstream.
cloud-down:
    terraform -chdir=infra destroy -auto-approve
