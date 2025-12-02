output "ecr_public_gpu_jobs_uri" {
  description = "ECR Public repository URI for gpu-jobs"
  value       = aws_ecrpublic_repository.gpu_jobs_public.repository_uri
}

output "ecr_public_gpu_base_uri" {
  description = "ECR Public repository URI for gpu-base"
  value       = aws_ecrpublic_repository.gpu_base_public.repository_uri
}


