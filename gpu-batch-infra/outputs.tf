output "ecr_repository_url" {
  description = "ECR repository URL for GPU jobs"
  value       = aws_ecr_repository.gpu_jobs.repository_url
}

output "ecr_base_image_url" {
  description = "ECR base image URL"
  value       = aws_ecr_repository.base_image.repository_url
}

output "batch_job_queue_name" {
  description = "Batch job queue name"
  value       = aws_batch_job_queue.gpu_queue.name
}

output "batch_job_definition_arn" {
  description = "Batch job definition ARN"
  value       = aws_batch_job_definition.gpu_generic.arn
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    raw         = aws_s3_bucket.raw_data.id
    finished    = aws_s3_bucket.finished_data.id
    checkpoints = aws_s3_bucket.checkpoints.id
  }
}

output "fsx_mount_command" {
  description = "FSx Lustre mount information"
  value = {
    dns_name   = aws_fsx_lustre_file_system.checkpoints.dns_name
    mount_name = aws_fsx_lustre_file_system.checkpoints.mount_name
    mount_path = "/fsx"
  }
}

output "cross_account_role_arn" {
  description = "Role ARN for team members to assume"
  value       = aws_iam_role.data_ops_contributor.arn
}

output "external_id" {
  description = "External ID for role assumption"
  value       = var.external_id
  sensitive   = true
}

output "team_instructions_nonsensitive" {
  value = <<-EOT
    aws sts assume-role \
      --role-arn ${aws_iam_role.data_ops_contributor.arn} \
      --role-session-name <your-alias> \
      --external-id <provided-out-of-band>
  EOT
}

