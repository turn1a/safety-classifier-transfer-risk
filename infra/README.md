# Cloud sweep

The cloud path runs the expensive TextAttack sweep on one ARM Graviton spot instance. Downstream transfer aggregation, target auditing, risk summaries, reporting, and MLflow run locally after the attack partitions are available. The infrastructure is plain Terraform and a bootstrap script.

The catalog owns where boundary artifacts live. Under `--env cloud`, eval splits, surrogate checkpoints, ONNX graphs, and adversarial partitions resolve to S3 through `${globals:...}`. The instance reads and writes those datasets through Kedro. Project data is not copied through ad hoc sync commands, and the instance does not need Hugging Face access.

## Default infrastructure and completed production run

Terraform defaults create a spot `c8g.16xlarge`, an ARM Graviton4 instance with 64 vCPUs. This is the current default configuration. It is not a description of the completed production sweep.

The completed production sweep used an ARM64 `r8g.48xlarge` Graviton4 spot instance with 192 vCPUs. Final shard reductions, target audit, risk calculations, figures, and reporting completed locally. The public results manifest records this completed-run provenance.

`terraform apply` creates an S3 exchange bucket, a least-privilege instance role and instance profile, an egress-only security group, and the configured spot instance. The box clones the configured repository ref, installs the cloud dependency group, runs the attack pipeline with `ParallelRunner` and `--only-missing-outputs`, then self-terminates.

## One-time AWS setup

1. Create a scoped IAM user rather than using the root account. Terraform needs permission to create and pass the instance role and profile.
1. Install Terraform and the AWS CLI.
1. Authenticate to gated Hugging Face models locally if the staging step needs them. The cloud instance itself has no Hugging Face token.
1. Copy `terraform.tfvars.example` to `terraform.tfvars` and set a globally unique bucket name plus the repository ref to execute.

Do not place credentials, account identifiers, bucket names, or local `terraform.tfvars` values in documentation or source control.

## Per-run workflow

```bash
just cloud-stage
just cloud-up
just cloud-logs
just cloud-finish
just cloud-down
```

`cloud-stage` creates the bucket and stages catalog boundary artifacts. `cloud-up` provisions the configured instance and starts the attack sweep. `cloud-finish` runs local downstream stages against the cloud partitions. `cloud-down` removes the created resources after downstream outputs are safe locally.

Resume is catalog-driven. `--only-missing-outputs` skips persisted partitions after an interruption or spot reclaim.

## Cost and sizing

The Terraform default is a sizing starting point, not a cost estimate for the completed production run. Spot prices, availability, runtime, region, selected instance type, and reruns determine cost. Review current AWS pricing and the Terraform plan before provisioning.

## Security controls

The instance role is scoped to the exchange bucket and SSM Session Manager core. Access uses SSM rather than an inbound SSH rule. The security group is egress-only and IMDSv2 is required. The staged instance holds no Hugging Face token and receives models through catalog-owned S3 artifacts.

## ONNX and torch behavior

Transformers export ONNX graphs in the model pipeline and retain them as parity checks. During the completed ARM64 production sweep, ONNX Runtime failed fused transformer attention with a matrix-dimension mismatch. Transformer victims therefore used torch checkpoints. The BiLSTM also uses torch.

An ONNX victim can be enabled only on a platform where the graph has been verified. The exported graphs remain useful for parity testing, while production ARM behavior is documented by the torch fallback.
