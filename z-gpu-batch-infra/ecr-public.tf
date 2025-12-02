resource "aws_ecrpublic_repository" "gpu_jobs_public" {
  repository_name = "gpu-jobs"

  tags = merge(local.common_tags, {
    Name = "Public GPU Jobs Container Registry"
  })
}

resource "aws_ecrpublic_repository" "gpu_base_public" {
  repository_name = "gpu-base"

  tags = merge(local.common_tags, {
    Name = "Public GPU Base Image Registry"
  })
}


