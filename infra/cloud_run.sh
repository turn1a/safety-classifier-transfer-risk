#!/usr/bin/env bash
# Cloud sweep runner — invoked on the EC2 box by user_data (as ec2-user). Installs the project
# and runs the attack sweep under `--env cloud`, reading inputs (splits, surrogate checkpoints,
# ONNX graphs) from S3 and writing each partition back to S3 *through the Kedro catalog* — no
# aws s3 sync, and no HuggingFace access (every model is materialised from the catalog).
# ParallelRunner uses the whole box; --only-missing-outputs makes a re-run (or a spot reclaim)
# resume by skipping partitions already on S3. Config is written to /opt/config.env by user_data:
# TR_BUCKET, TR_REGION, TR_REPO_REF.
set -euo pipefail
# Export the box config into the environment so the catalog's ${tr.bucket:} / ${tr.region:}
# resolvers and the AWS SDK (region) see it.
set -a
source /opt/config.env
set +a
export AWS_REGION="$TR_REGION"
export AWS_DEFAULT_REGION="$TR_REGION"

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DO_NOT_TRACK=1
# ParallelRunner start method: spawn (not the Linux default fork). Each attack node sets
# TA_DEVICE / OMP_NUM_THREADS and imports textattack/torch inside the worker, which only takes
# effect in a fresh process; spawn also avoids fork-after-torch issues (Kedro loads every
# pipeline module — and thus torch — in the parent). The node payloads are spawn-safe
# (module-level functions + functools.partial; all datasets persisted).
export KEDRO_MP_CONTEXT=spawn

cd "$HOME/repo"
git checkout "$TR_REPO_REF"

# uv + the project, including the cloud group (s3fs) so the catalog can read/write s3://.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group cloud

# Offline NLP assets the recipes need (NLTK data, counter-fitted embeddings, sentence encoder).
uv run python -m transfer_risk.scripts.fetch_assets

# The sweep, in tmux so `just cloud-attach` can watch it live. The catalog reads inputs from S3
# and writes each shard/cell partition straight to S3; --only-missing-outputs resumes. --async
# loads a node's inputs (splits + victim from S3) and saves its partition on threads, overlapping
# the per-shard S3 I/O with compute. NodeTimingHook logs each node's load/compute/save time.
tmux new-session -d -s run \
  "uv run kedro run --env cloud --pipeline attacks --runner ParallelRunner --async --only-missing-outputs; echo done > /tmp/sweep_done"
while [ ! -f /tmp/sweep_done ]; do sleep 15; done
