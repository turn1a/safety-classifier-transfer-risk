output "bucket_name" {
  description = "The state-exchange bucket."
  value       = aws_s3_bucket.exchange.id
}

output "instance_id" {
  description = "The spot instance id."
  value       = aws_instance.runner.id
}

output "ssm_session_command" {
  description = "Open a shell on the box (no SSH or key pair needed)."
  value       = "aws ssm start-session --target ${aws_instance.runner.id} --region ${var.region} --profile ${var.aws_profile}"
}

output "log_tail_command" {
  description = "Stream the bootstrap/run log from S3."
  value       = "aws s3 cp s3://${aws_s3_bucket.exchange.id}/logs/latest/run.log - --region ${var.region} --profile ${var.aws_profile}"
}
