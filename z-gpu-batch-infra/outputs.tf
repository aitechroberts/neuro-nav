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
    datasets    = aws_s3_bucket.datasets.id
  }
}

# COMMENTED OUT - FSx deleted to save costs. Uncomment when ready to use.
# output "fsx_mount_command" {
#   description = "FSx Lustre mount information"
#   value = {
#     dns_name   = aws_fsx_lustre_file_system.checkpoints.dns_name
#     mount_name = aws_fsx_lustre_file_system.checkpoints.mount_name
#     mount_path = "/fsx"
#   }
# }

# output "fsx_datasets_mount_command" {
#   description = "FSx Lustre SCRATCH mount information for datasets"
#   value = {
#     dns_name   = aws_fsx_lustre_file_system.datasets.dns_name
#     mount_name = aws_fsx_lustre_file_system.datasets.mount_name
#     mount_path = "/fsx-datasets"
#     s3_bucket  = aws_s3_bucket.datasets.bucket
#   }
# }

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

# ==========================================
# ECS Fargate Data UI Outputs
# ==========================================

output "data_ui_cluster" {
  description = "ECS cluster name for data-ui"
  value       = aws_ecs_cluster.data_ui.name
}

output "data_ui_service" {
  description = "ECS service name for data-ui"
  value       = aws_ecs_service.data_ui.name
}

output "data_ui_ecr_repository" {
  description = "ECR repository URL for data-ui image"
  value       = aws_ecr_repository.data_ui.repository_url
}

output "data_ui_scale_commands" {
  description = "Commands to scale the data-ui service up/down"
  value = <<-EOT
    # Scale UP (start the UI):
    aws ecs update-service --cluster ${aws_ecs_cluster.data_ui.name} --service ${aws_ecs_service.data_ui.name} --desired-count 1

    # Scale DOWN (stop to save money):
    aws ecs update-service --cluster ${aws_ecs_cluster.data_ui.name} --service ${aws_ecs_service.data_ui.name} --desired-count 0

    # Get public IP (after scaling up):
    aws ecs list-tasks --cluster ${aws_ecs_cluster.data_ui.name} --service ${aws_ecs_service.data_ui.name} --query 'taskArns[0]' --output text | xargs -I {} aws ecs describe-tasks --cluster ${aws_ecs_cluster.data_ui.name} --tasks {} --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} --query 'NetworkInterfaces[0].Association.PublicIp' --output text
  EOT
}

