# Cloud sweep (infra/)

The attack sweep is the only expensive stage of the pipeline. This directory runs it on a single high-core ARM Graviton **spot** instance, where the cheap downstream (transfer, risk, reporting) and MLflow run locally afterwards. It is plain Terraform plus a bootstrap script — no AWS Batch, no Kubernetes, no distributed executor.

The catalog owns where data lives. Under `--env cloud` the boundary artifacts (eval splits, surrogate checkpoints, ONNX graphs, adversarial partitions) resolve to the S3 bucket through `${globals:...}` (see `conf/cloud/globals.yml`), so the box reads inputs and writes partitions **through the Kedro catalog** — there is no `aws s3 sync` of project data, and the box never touches the HuggingFace Hub.

## What gets created

`terraform apply` creates, in your account and region: an S3 bucket (the state exchange — surrogates, ONNX graphs, splits, partitions, logs), a least-privilege IAM role and instance profile for the box (read/write that one bucket + SSM Session Manager core), an egress-only security group (shell access is via SSM Session Manager, so there is no SSH and no inbound rule), and a spot `c8g.16xlarge` (64 vCPU Graviton4) with IMDSv2 required. On boot the box clones this repo at the pinned ref, installs the project (`uv sync --group cloud`), runs `kedro run --env cloud --pipeline attacks --runner ParallelRunner --only-missing-outputs` (reading inputs and writing partitions via the catalog), and self-terminates.

## One-time AWS setup

This is the entire manual surface. After it, the `just cloud-*` recipes are one word each.

1. **A scoped IAM user (do not use the root user).** In the IAM console create a user (e.g. `transfer-risk-cli`, no console access), attach `PowerUserAccess`, and add an inline policy for the IAM actions PowerUser excludes (Terraform creates the instance role and passes it to EC2):
   ```json
   { "Version": "2012-10-17", "Statement": [{ "Effect": "Allow", "Resource": "*",
     "Action": ["iam:CreateRole","iam:DeleteRole","iam:GetRole","iam:GetRolePolicy","iam:PassRole",
       "iam:PutRolePolicy","iam:DeleteRolePolicy","iam:AttachRolePolicy","iam:DetachRolePolicy",
       "iam:CreateInstanceProfile","iam:DeleteInstanceProfile","iam:GetInstanceProfile",
       "iam:AddRoleToInstanceProfile","iam:RemoveRoleFromInstanceProfile",
       "iam:ListInstanceProfilesForRole","iam:ListRolePolicies","iam:ListAttachedRolePolicies"] }]}
   ```
   Create an access key for it ("Command Line Interface"), then `aws configure --profile transfer-risk-cli` (paste the key/secret, region `eu-central-1`, output `json`). Delete any **root** access keys. Verify: `aws sts get-caller-identity --profile transfer-risk-cli` shows `.../user/transfer-risk-cli`, not `:root`.
1. **Install Terraform** (the recipes call it): `brew install terraform` (or `brew install hashicorp/tap/terraform`). The AWS CLI is also required (log streaming, SSM shell).
1. **HuggingFace auth — local only.** The box needs no token (every model is loaded from the catalog). `cloud-stage` trains the pool locally and downloads the gated models, so authenticate on your machine: `huggingface-cli login`, or set `HF_TOKEN` in `.env`. Request access on each gated model's HF page first.
1. **Fill `terraform.tfvars`** (copy from `terraform.tfvars.example`): a globally-unique `bucket_name` and the `repo_ref` (the git SHA to run). Everything else has defaults.

## Per-run workflow (from the repo root)

```bash
just cloud-stage     # create the bucket; train the pool once under --env cloud, staging inputs (splits, surrogates, ONNX) to S3
just cloud-up        # provision the box; it runs the sweep (ParallelRunner) and self-terminates (~<1h)
just cloud-logs      # stream the box's log from S3 (or: just cloud-attach, then `tmux attach -t run`)
just cloud-finish    # downstream (transfer/risk/reporting) + MLflow, locally, reading partitions from S3
just cloud-down      # tear everything down (run cloud-finish first; results are local by then)
```

Iterating (tweak params, re-run) costs only another `cloud-up` → `cloud-finish`; the box self-terminates after each run, so nothing bills between runs except a few cents of S3 storage. Resume is automatic: `--only-missing-outputs` skips partitions already on S3, so a re-run (or a spot reclaim mid-sweep) only attacks what is missing. `cloud-stage` is a single training pass; re-running it skips surrogates already on S3.

## Cost, security, and the ONNX fallback

Cost is roughly **$1-2 per full run**: a 64-vCPU Graviton spot box at ~$0.8-1.3/hr finishing in under an hour, plus a few cents of same-region S3. Spot interruption is safe — each cell's partition is written to S3 the moment it completes (a single object PUT), so a reclaim costs only the in-flight shards and `--only-missing-outputs` resumes. Set `use_spot = false` for on-demand if a region is capacity-constrained.

Security: the instance role can read/write only this one bucket (plus SSM Session Manager core); access is via SSM Session Manager (IAM-gated, audited, no open port); IMDSv2 is required. The box holds no HuggingFace token and makes no Hub calls — all models arrive from the catalog (S3).

ONNX fallback: transformer victims are served from their ONNX graphs (exported in-pipeline by a `torch.onnx.export` node and validated against the torch checkpoint by the parity gate); the BiLSTM stays on torch. ONNX Runtime is validated on arm64 first. If arm64 Linux `onnxruntime` wheels or parity misbehave, set `arch = "x86_64"` and `instance_type = "c7i.16xlarge"` in `terraform.tfvars` and re-`apply` (x86_64 ONNX wheels are the most mature); the exported graphs are portable and the rest of the pipeline is arch-agnostic.
