aws_region   = "us-east-1"
environment  = "dev"
project_name = "gpu-batch"

notification_email = "jrobert@andrew.cmu.edu"

team_accounts = {
  "jesse" = "677748260524",
  "abdul" = "995726271638",
  "nick"  = "509399632158",
}

iam_user_names = ["jabarkle","nchermak"]

iam_group_name = "TalkingRobots"

external_id = "gpu-batch-fall-2025-secret"

gpu_instance_types = ["g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g4dn.xlarge", "g4dn.2xlarge"]
max_vcpus          = 256
bid_percentage     = 100
