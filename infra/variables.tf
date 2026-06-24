variable "region" {
  description = "AWS region for the bucket and the spot instance (keep them the same to avoid cross-region transfer)."
  type        = string
  default     = "eu-central-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile Terraform and the cloud-* recipes use."
  type        = string
  default     = "transfer-risk-cli"
}

variable "bucket_name" {
  description = "Globally-unique S3 bucket for the state exchange (surrogates, data, partitions, logs)."
  type        = string
}

variable "repo_url" {
  description = "Public git URL the cloud box clones."
  type        = string
  default     = "https://github.com/turn1a/safety-classifier-transfer-risk.git"
}

variable "repo_ref" {
  description = "Git ref (branch or SHA) the cloud box checks out — pin a SHA for a reproducible run."
  type        = string
}

variable "arch" {
  description = "CPU architecture: arm64 (Graviton, default) or x86_64 (the ONNX fallback)."
  type        = string
  default     = "arm64"
  validation {
    condition     = contains(["arm64", "x86_64"], var.arch)
    error_message = "arch must be arm64 or x86_64."
  }
}

variable "instance_type" {
  description = "EC2 instance type; must match arch (arm64 -> c8g/c7g.16xlarge, x86_64 -> c7i/c7a.16xlarge)."
  type        = string
  default     = "c8g.16xlarge"
}

variable "availability_zone" {
  description = "AZ to launch the instance in. Pin to the cheapest spot AZ — spot price varies by AZ within a region (r8g.48xlarge: 1c/1a ~$2.2 vs 1b ~$4.0)."
  type        = string
  default     = "eu-central-1c"
}

variable "use_spot" {
  description = "Use a spot instance (cheapest; the sweep is resumable so an interruption is safe)."
  type        = bool
  default     = true
}

variable "root_volume_gb" {
  description = "Root EBS size (holds the venv, the HF cache, and a working copy of data/)."
  type        = number
  default     = 100
}

variable "max_lifetime_minutes" {
  description = "Safety auto-shutdown so a hung or idle run cannot bill indefinitely."
  type        = number
  default     = 180
}
