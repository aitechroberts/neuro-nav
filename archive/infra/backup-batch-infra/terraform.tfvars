aws_region   = "us-east-1"
environment  = "dev"
project_name = "gpu-batch"

notification_email = "jrobert@andrew.cmu.edu"

team_accounts = {
  "main_account" = "585780419748"
}

iam_user_names = ["gmetts","ewassman", "samyakt"]

iam_group_name = "VLR-Project"

external_id = "gpu-batch-fall-2025-secret"

gpu_instance_types = ["g5.xlarge", "g5.2xlarge", "g4dn.xlarge", "g4dn.2xlarge"]
max_vcpus          = 8
bid_percentage     = 100
