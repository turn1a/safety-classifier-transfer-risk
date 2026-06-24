provider "aws" {
  region  = var.region
  profile = var.aws_profile
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  # Pin to the chosen AZ so the (spot) instance lands in the cheapest subzone — spot price varies by AZ.
  filter {
    name   = "availability-zone"
    values = [var.availability_zone]
  }
}

# Always the newest Amazon Linux 2023 AMI for the chosen arch — no hard-coded, expiring id.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-${var.arch}"
}

# --- S3: the state-exchange bucket (surrogates, data, partitions, logs) ----
resource "aws_s3_bucket" "exchange" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_versioning" "exchange" {
  bucket = aws_s3_bucket.exchange.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "exchange" {
  bucket = aws_s3_bucket.exchange.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "exchange" {
  bucket                  = aws_s3_bucket.exchange.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "exchange" {
  bucket = aws_s3_bucket.exchange.id
  rule {
    id     = "abort-incomplete-mpu"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# --- IAM: least-privilege instance role ------------------------------------
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runner" {
  name               = "transfer-risk-runner"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# Session Manager (shell access without SSH / an inbound rule / a key pair).
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "runner" {
  statement {
    sid       = "BucketObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.exchange.arn}/*"]
  }
  statement {
    sid       = "BucketList"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.exchange.arn]
  }
  # No HuggingFace-token statement: the box reads every model from the catalog (S3), so it
  # never calls the Hub and needs no SSM/KMS access beyond AmazonSSMManagedInstanceCore.
}

resource "aws_iam_role_policy" "runner" {
  name   = "transfer-risk-runner"
  role   = aws_iam_role.runner.id
  policy = data.aws_iam_policy_document.runner.json
}

resource "aws_iam_instance_profile" "runner" {
  name = "transfer-risk-runner"
  role = aws_iam_role.runner.name
}

# --- Security group: egress only (inbound access is via SSM Session Manager) ---
resource "aws_security_group" "runner" {
  name        = "transfer-risk-runner"
  description = "Egress only; shell access via SSM Session Manager (no SSH, no inbound rule)."
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All outbound (GitHub, PyPI, HuggingFace, S3, SSM endpoints)."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- Spot EC2 runner -------------------------------------------------------
resource "aws_instance" "runner" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.runner.name
  vpc_security_group_ids = [aws_security_group.runner.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  # Spot is the default (cheapest); one-time + terminate suits a fire-and-forget, resumable
  # sweep. Set use_spot = false for on-demand if a region is spot-capacity-constrained.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    throughput            = 250
    delete_on_termination = true
  }

  # IMDSv2 required, single hop: the instance runs the pipeline as a host process, not a
  # container, so the role credentials cannot be reached from an SSRF against IMDSv1.
  metadata_options {
    http_tokens                 = "required"
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    bucket       = aws_s3_bucket.exchange.id
    region       = var.region
    repo_url     = var.repo_url
    repo_ref     = var.repo_ref
    max_lifetime = var.max_lifetime_minutes
  })
  # Editing the bootstrap recreates the instance (user-data runs only at first boot).
  user_data_replace_on_change = true

  tags = {
    Project = "transfer-risk"
    Role    = "sweep-runner"
  }
}
