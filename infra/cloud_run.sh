#!/usr/bin/env bash
# Cloud sweep runner — invoked on the EC2 box by user_data (as ec2-user). Installs the project
# and runs the attack sweep under `--env cloud`, reading inputs (splits, surrogate checkpoints,
# ONNX graphs) from S3 and writing each partition back to S3 *through the Kedro catalog* — no
# aws s3 sync, and no HuggingFace access (every model is materialised from the catalog).
# RamBoundedParallelRunner runs one worker per core that fits in RAM (each torch-victim worker needs
# several GB, so one-per-vCPU OOMs a memory-light box); --only-missing-outputs makes a re-run (or a
# spot reclaim) resume by skipping partitions already on S3. Config is written to /opt/config.env by
# user_data: TR_BUCKET, TR_REGION, TR_REPO_REF.
set -euo pipefail
# Export the box config into the environment so the catalog's ${tr.bucket:} / ${tr.region:}
# resolvers and the AWS SDK (region) see it.
set -a
source /opt/config.env
set +a
export AWS_REGION="$TR_REGION"
export AWS_DEFAULT_REGION="$TR_REGION"
# Select the cloud config environment. The catalog's S3 routing keys off the run env, and in this
# project the KEDRO_ENV variable is honoured where `kedro run --env cloud` alone is not (the
# catalog otherwise resolves the boundary datasets to their local base paths), so set it here.
export KEDRO_ENV=cloud

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DO_NOT_TRACK=1
# Pin every inner parallelism to one: the runner already runs one worker per core (capped to fit
# RAM). Nested BLAS threads or joblib/loky process pools (textattack's embedding search spawns one
# per worker, sized to the CPU count) would oversubscribe and exhaust processes/RAM — the
# BrokenProcessPool that aborted an earlier sweep. Set here so each spawn worker inherits it before
# importing numpy/torch (some of these are read only at import time).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
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

# The sweep, run in the foreground so its output flows through user_data's `tee` to the S3 log
# (cloud-logs) — the NodeTimingHook's per-node load/compute/save lines and any traceback are then
# visible without attaching (running it detached in tmux hid all of this from S3). The catalog
# reads inputs from S3 and writes each shard/cell partition straight to S3; --only-missing-outputs
# resumes; --async overlaps the per-shard S3 I/O with compute. `|| true` lets user_data do its
# final log sync + shutdown even if the run errors.
uv run kedro run --env cloud --pipeline attacks \
  --runner transfer_risk.runner.RamBoundedParallelRunner --async --only-missing-outputs || true
