# Cloud sweep (infra/)

The attack sweep is the only expensive stage of the pipeline. This directory runs it on a single high-core ARM Graviton **spot** instance and brings the results back to your machine, where the cheap downstream (transfer, risk, reporting) and MLflow run locally. It is plain Terraform plus a bootstrap script — no AWS Batch, no Kubernetes, no distributed executor.

## What gets created

`terraform apply` creates, in your account and region: an S3 bucket (the state exchange — surrogates, data, partitions, logs), a least-privilege IAM role and instance profile for the box, an egress-only security group (shell access is via SSM Session Manager, so there is no SSH and no inbound rule), and a spot `c8g.16xlarge` (64 vCPU Graviton4) with IMDSv2 required. On boot the box clones this repo at the pinned ref, syncs the cached inputs from S3, runs `kedro run --env cloud --pipeline attacks`, syncs the per-cell partitions back to S3 as they complete, and self-terminates.

## One-time AWS setup

This is the entire manual surface. After it, the `just cloud-*` recipes are one word each.

1. **A scoped IAM user (do not use the root user).** In the IAM console create a user (e.g. `transfer-risk-cli`, no console access), attach `PowerUserAccess`, and add an inline policy for the IAM actions PowerUser excludes (Terraform creates the instance role and passes it to EC2):
   ```json
   { "Version": "2012-10-17", "Statement": [{ "Effect": "Allow", "Resource": "*",
     "Action": ["iam:CreateRole","iam:DeleteRole","iam:GetRole","iam:PassRole",
       "iam:PutRolePolicy","iam:DeleteRolePolicy","iam:AttachRolePolicy","iam:DetachRolePolicy",
       "iam:CreateInstanceProfile","iam:DeleteInstanceProfile","iam:GetInstanceProfile",
       "iam:AddRoleToInstanceProfile","iam:RemoveRoleFromInstanceProfile",
       "iam:ListInstanceProfilesForRole","iam:ListRolePolicies","iam:ListAttachedRolePolicies"] }]}
   ```
   Create an access key for it ("Command Line Interface"), then `aws configure --profile transfer-risk-cli` (paste the key/secret, region `eu-central-1`, output `json`). Delete any **root** access keys. Verify: `aws sts get-caller-identity --profile transfer-risk-cli` shows `.../user/transfer-risk-cli`, not `:root`.
1. **Install Terraform** (the recipes call it): `brew install terraform` (or `brew install hashicorp/tap/terraform`). The AWS CLI is already required for the sync recipes.
1. **Store the HuggingFace token** (gated models + to avoid download throttling), encrypted at rest:
   ```bash
   aws ssm put-parameter --name /transfer-risk/hf-token --type SecureString \
     --value "hf_xxxxxxxx" --region eu-central-1 --profile transfer-risk-cli --overwrite
   ```
1. **Fill `terraform.tfvars`** (copy from `terraform.tfvars.example`): a globally-unique `bucket_name` and the `repo_ref` (the git SHA to run). Everything else has defaults.

## Per-run workflow (from the repo root)

```bash
just export-onnx     # export surrogates to ONNX (optional; ~2-3x faster victim queries)
just cloud-push      # upload surrogates + canonical data + splits to the bucket
just cloud-up        # provision the box; it runs the sweep and self-terminates (~<1h)
just cloud-logs      # stream the box's log from S3 (or: just cloud-attach, then `tmux attach -t run`)
just cloud-pull      # bring the adversarial partitions back into data/
just run --from-nodes evaluate_transfer   # downstream + MLflow, locally
just cloud-down      # tear everything down (run cloud-pull first; results are local by then)
```

Iterating (tweak data/models/params, re-run) costs only another `cloud-up` → `cloud-pull`; the box self-terminates after each run, so nothing bills between runs except a few cents of S3 storage. Resume is automatic: completed cells are skipped, so a re-run (or a spot reclaim mid-sweep) only attacks what is missing.

## Cost, security, and the ONNX fallback

Cost is roughly **$1-2 per full run**: a 64-vCPU Graviton spot box at ~$0.8-1.3/hr finishing in under an hour, plus a few cents of same-region S3. Spot interruption is safe — partitions stream to S3 every ~90 s, so a reclaim costs only the in-flight cells. Set `use_spot = false` for on-demand if a region is capacity-constrained.

Security: the instance role can read/write only this one bucket and read only the one HF-token SSM parameter; access is via SSM Session Manager (IAM-gated, audited, no open port); IMDSv2 is required; the HF token is fetched into the run's process environment only, never written to disk or logged.

ONNX fallback: ONNX is validated on arm64 first. If arm64 Linux `onnxruntime` wheels or parity misbehave, set `arch = "x86_64"` and `instance_type = "c7i.16xlarge"` in `terraform.tfvars` and re-`apply` (x86_64 ONNX wheels are the most mature); the exported graphs are portable and the rest of the pipeline is arch-agnostic. `use_onnx: false` (in `conf/cloud/parameters_attacks.yml`) always restores the pure-torch path.
