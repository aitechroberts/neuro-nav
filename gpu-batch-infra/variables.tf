variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev/prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "gpu-batch"
}


variable "team_accounts" {
  description = "Map of team member aliases to their AWS account IDs"
  type        = map(string)
  default = {
    # Add team members here
    # "alice" = "111111111111"
    # "bob"   = "222222222222"
  }
}

variable "external_id" {
  description = "External ID for cross-account role assumption"
  type        = string
  default     = "gpu-batch-class-2024"
  sensitive   = true
}

variable "notification_email" {
  description = "Email for job failure notifications"
  type        = string
}

variable "gpu_instance_types" {
  description = "GPU instance types for Batch"
  type        = list(string)
  default     = ["g5.4xlarge", "g5.8xlarge"]
}

variable "max_vcpus" {
  description = "Maximum vCPUs for Batch compute environment"
  type        = number
  default     = 256
}

variable "bid_percentage" {
  description = "Spot bid percentage"
  type        = number
  default     = 100
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GB (1200 or increments of 2400)"
  type        = number
  default     = 2400
}

variable "fsx_throughput" {
  description = "FSx Lustre throughput in MB/s/TiB"
  type        = number
  default     = 500

  validation {
    condition     = contains([125, 250, 500, 1000], var.fsx_throughput)
    error_message = "Throughput must be 125, 250, 500, or 1000 MB/s/TiB"
  }
}