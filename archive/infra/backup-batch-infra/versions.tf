terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Optional: Store state in S3
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "gpu-batch/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  profile = "acct1"
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GPU-Batch-Processing"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

