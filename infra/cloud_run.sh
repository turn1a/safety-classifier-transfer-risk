#!/usr/bin/env bash
# Cloud sweep runner — invoked on the EC2 box by user_data (as ec2-user). Installs the project,
# pulls cached inputs (and any partitions a previous run finished) from S3, runs the attack
# sweep in the cloud env (all cores, ONNX victims, small shards), and pushes completed
# partitions back to S3 — incrementally, so a spot reclaim loses at most the in-flight cells.
# Config is written to /opt/config.env by user_data: TR_BUCKET, TR_REGION, TR_REPO_REF,
# TR_SSM_TOKEN_PARAM. This is a plain bash file (not a Terraform template), so $VARS are normal.
set -euo pipefail
source /opt/config.env

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DO_NOT_TRACK=1

cd "$HOME/repo"
git checkout "$TR_REPO_REF"

# uv + the project (resolves torch/transformers/textattack/onnxruntime arm64 wheels from the lock).
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Offline NLP assets the recipes need (NLTK data, counter-fitted embeddings, sentence encoder).
uv run python -m transfer_risk.scripts.fetch_assets

# HuggingFace token into the process env only — never written to disk or logged.
HF_TOKEN="$(aws ssm get-parameter --name "$TR_SSM_TOKEN_PARAM" --with-decryption \
  --region "$TR_REGION" --query Parameter.Value --output text)"
export HF_TOKEN

# Pull cached inputs (surrogates incl. ONNX + BiLSTM + manifest, canonical data, splits) and
# any partitions a previous run already completed (resume skips them).
aws s3 sync "s3://$TR_BUCKET/data/03_primary/" data/03_primary/ --region "$TR_REGION"
aws s3 sync "s3://$TR_BUCKET/data/05_model_input/" data/05_model_input/ --region "$TR_REGION"
aws s3 sync "s3://$TR_BUCKET/data/06_models/" data/06_models/ --region "$TR_REGION"
aws s3 sync "s3://$TR_BUCKET/data/07_model_output/adversarial_examples/" \
  data/07_model_output/adversarial_examples/ --region "$TR_REGION"

# Push completed partitions every 90s so a spot reclaim costs only the in-flight cells.
( while true; do
    aws s3 sync data/07_model_output/adversarial_examples/ \
      "s3://$TR_BUCKET/data/07_model_output/adversarial_examples/" --region "$TR_REGION" \
      >/dev/null 2>&1
    sleep 90
  done ) &
sync_pid=$!

# The sweep, in tmux so `just cloud-attach` can watch it live. The trailing marker fires whether
# the run succeeds or fails, so partial results are still pushed below.
tmux new-session -d -s run \
  "uv run kedro run --env cloud --pipeline attacks; echo done > /tmp/sweep_done"
while [ ! -f /tmp/sweep_done ]; do sleep 15; done

kill "$sync_pid" 2>/dev/null || true
aws s3 sync data/07_model_output/adversarial_examples/ \
  "s3://$TR_BUCKET/data/07_model_output/adversarial_examples/" --region "$TR_REGION"
