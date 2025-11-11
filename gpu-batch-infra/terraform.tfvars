aws_region   = "us-east-1"
environment  = "dev"
project_name = "gpu-batch"

notification_email = "jrobert+11851@andrew.cmu.edu"

team_accounts = {
  "jesse" = "677748260524",
  "abdul" = "995726271638",
  "nick"  = "509399632158",
}

iam_user_names = ["jabarkle","nchermak"]

iam_group_name = "TalkingRobots"

external_id = "gpu-batch-fall-2025-secret"

gpu_instance_types = ["g5.4xlarge", "g5.8xlarge"]
max_vcpus          = 256
bid_percentage     = 100

fsx_storage_capacity = 2400
fsx_throughput       = 500