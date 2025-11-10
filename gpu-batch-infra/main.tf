data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {}

locals {
  account_id    = data.aws_caller_identity.current.account_id
  bucket_suffix = "${local.account_id}-${var.aws_region}" # or: format("%s-%s", local.account_id, var.aws_region)

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets # use private subnets for Batch/FSx
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}


module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["${var.aws_region}a", "${var.aws_region}b"]

  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.11.0/24", "10.0.12.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}



# ==========================================
# S3 Buckets
# ==========================================

resource "aws_s3_bucket" "raw_data" {
  bucket = "data-raw-${local.bucket_suffix}"

  tags = merge(local.common_tags, {
    Name = "Raw Data Bucket"
    Type = "Input"
  })
}

resource "aws_s3_bucket" "finished_data" {
  bucket = "data-finished-${local.bucket_suffix}"

  tags = merge(local.common_tags, {
    Name = "Finished Data Bucket"
    Type = "Output"
  })
}

resource "aws_s3_bucket" "checkpoints" {
  bucket = "model-checkpoints-${local.bucket_suffix}"

  tags = merge(local.common_tags, {
    Name = "Model Checkpoints"
    Type = "Checkpoints"
  })
}

# Bucket ownership controls
resource "aws_s3_bucket_ownership_controls" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_ownership_controls" "finished_data" {
  bucket = aws_s3_bucket.finished_data.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_ownership_controls" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ==========================================
# ECR Repository
# ==========================================

resource "aws_ecr_repository" "gpu_jobs" {
  name                 = "gpu-jobs"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.common_tags, {
    Name = "GPU Jobs Container Registry"
  })
}

# Base image repository
resource "aws_ecr_repository" "base_image" {
  name                 = "gpu-base"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(local.common_tags, {
    Name = "Base Image for Team"
  })
}

resource "aws_ecr_repository_policy" "gpu_jobs_cross_account" {
  repository = aws_ecr_repository.gpu_jobs.name
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Sid : "AllowTeamPushPull",
      Effect : "Allow",
      Principal = {
        AWS = [for alias, account_id in var.team_accounts : "arn:aws:iam::${account_id}:root"]
      },
      Action = [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:CompleteLayerUpload",
        "ecr:GetDownloadUrlForLayer",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      #   Resource = "*"
    }]
  })
}

# ==========================================
# FSx Lustre for Checkpoints
# ==========================================

resource "aws_security_group" "fsx_lustre" {
  name_prefix = "${var.project_name}-fsx-"
  vpc_id      = local.vpc_id
  description = "Security group for FSx Lustre"

  # FSx server <-> clients (Batch instances) on Lustre ports
  ingress {
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = [aws_security_group.batch_compute.id]
  }

  ingress {
    from_port       = 1018
    to_port         = 1023
    protocol        = "tcp"
    security_groups = [aws_security_group.batch_compute.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-fsx-sg"
  })
}

# Self-referencing rules (FSx servers ↔ FSx servers)
resource "aws_security_group_rule" "fsx_self_988" {
  type                     = "ingress"
  from_port                = 988
  to_port                  = 988
  protocol                 = "tcp"
  security_group_id        = aws_security_group.fsx_lustre.id
  source_security_group_id = aws_security_group.fsx_lustre.id
}

resource "aws_security_group_rule" "fsx_self_1018_1023" {
  type                     = "ingress"
  from_port                = 1018
  to_port                  = 1023
  protocol                 = "tcp"
  security_group_id        = aws_security_group.fsx_lustre.id
  source_security_group_id = aws_security_group.fsx_lustre.id
}

data "aws_vpc" "selected" {
  id = local.vpc_id
}

resource "aws_fsx_lustre_file_system" "checkpoints" {
  storage_capacity            = var.fsx_storage_capacity
  subnet_ids                  = [local.subnet_ids[0]]
  deployment_type             = "PERSISTENT_2"
  per_unit_storage_throughput = var.fsx_throughput # 125|250|500|1000

  security_group_ids = [aws_security_group.fsx_lustre.id]
  # Optional DRAs instead of inline import/export for more control (future)
  # See AWS docs if you later add aws_fsx_data_repository_association

  tags = merge(local.common_tags, { Name = "${var.project_name}-fsx-checkpoints" })
}

# ==========================================
# IAM Roles for Batch
# ==========================================

# Batch Service Role
resource "aws_iam_role" "batch_service" {
  name = "${var.project_name}-batch-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "batch.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# Spot Fleet Role
resource "aws_iam_role" "spot_fleet" {
  name = "${var.project_name}-spot-fleet-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "spotfleet.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# ECS Task Execution Role
resource "aws_iam_role" "batch_execution" {
  name = "${var.project_name}-batch-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "batch_execution" {
  role       = aws_iam_role.batch_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role (for containers)
resource "aws_iam_role" "batch_job" {
  name = "${var.project_name}-batch-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

# Job role policies
resource "aws_iam_role_policy" "batch_job_s3" {
  name = "s3-access"
  role = aws_iam_role.batch_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.raw_data.arn,
          "${aws_s3_bucket.raw_data.arn}/*",
          aws_s3_bucket.checkpoints.arn,
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.finished_data.arn}/*",
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "batch_job_fsx" {
  name = "fsx-access"
  role = aws_iam_role.batch_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "fsx:DescribeFileSystems",
        "fsx:DescribeDataRepositoryTasks"
      ]
      Resource = aws_fsx_lustre_file_system.checkpoints.arn
    }]
  })
}

# ==========================================
# Cross-Account Access Role
# ==========================================

resource "aws_iam_role" "data_ops_contributor" {
  name = "DataOpsContributor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        AWS = [for alias, account_id in var.team_accounts :
        "arn:aws:iam::${account_id}:root"]
      }
      Action = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "sts:ExternalId" = var.external_id
        }
      }
    }]
  })

  tags = merge(local.common_tags, {
    Name = "Cross-Account Team Access"
  })
}

resource "aws_iam_role_policy" "data_ops_contributor" {
  name = "contributor-permissions"
  role = aws_iam_role.data_ops_contributor.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action : ["ecr:GetAuthorizationToken"],
        Resource : "*"
      },
      # Repo-scoped push/pull
      {
        Sid : "ECRPushPull",
        Effect : "Allow",
        Action : [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ],
        Resource : [
          aws_ecr_repository.gpu_jobs.arn,
          aws_ecr_repository.base_image.arn
        ]
      },
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.raw_data.arn}/*",
          "${aws_s3_bucket.checkpoints.arn}/*",
          aws_s3_bucket.raw_data.arn,
          aws_s3_bucket.checkpoints.arn
        ]
      }
    ]
  })
}

# ==========================================
# Batch Compute Environment
# ==========================================

resource "aws_security_group" "batch_compute" {
  name_prefix = "${var.project_name}-batch-"
  vpc_id      = local.vpc_id
  description = "Security group for Batch compute instances"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-batch-compute-sg"
  })
}

resource "aws_launch_template" "batch_gpu" {
  name_prefix = "${var.project_name}-gpu-"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  # User data to mount FSx Lustre
  user_data = base64encode(templatefile("${path.module}/modules/user_data.sh", {
    fsx_dns_name   = aws_fsx_lustre_file_system.checkpoints.dns_name
    fsx_mount_name = aws_fsx_lustre_file_system.checkpoints.mount_name
  }))

  tags = local.common_tags
}

resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "${var.project_name}-gpu-spot"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    bid_percentage      = var.bid_percentage
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn

    min_vcpus     = 0
    desired_vcpus = 0
    max_vcpus     = var.max_vcpus

    instance_type = var.gpu_instance_types

    subnets = local.subnet_ids

    security_group_ids = [aws_security_group.batch_compute.id]

    instance_role = aws_iam_instance_profile.batch_instance.arn

    launch_template {
      launch_template_id = aws_launch_template.batch_gpu.id
      version            = "$Latest"
    }

    tags = merge(local.common_tags, {
      Name = "${var.project_name}-batch-gpu-instance"
    })
  }

  tags = local.common_tags
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.project_name}-batch-instance-profile"
  role = aws_iam_role.batch_instance.name
}

resource "aws_iam_role" "batch_instance" {
  name = "${var.project_name}-batch-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# ==========================================
# Batch Job Queue and Definition
# ==========================================

resource "aws_batch_job_queue" "gpu_queue" {
  name     = "${var.project_name}-gpu-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }

  tags = local.common_tags
}

resource "aws_batch_job_definition" "gpu_generic" {
  name = "${var.project_name}-gpu-generic"
  type = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image = "${aws_ecr_repository.gpu_jobs.repository_url}:latest"

    command = ["bash", "-c", "echo 'Override this command'"]

    jobRoleArn       = aws_iam_role.batch_job.arn
    executionRoleArn = aws_iam_role.batch_execution.arn

    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      },
      {
        type  = "VCPU"
        value = "4"
      },
      {
        type  = "MEMORY"
        value = "30000"
      }
    ]

    environment = [
      {
        name  = "FSX_MOUNT"
        value = "/fsx"
      }
    ]

    mountPoints = [
      {
        containerPath = "/fsx"
        sourceVolume  = "fsx-lustre"
      }
    ]

    volumes = [
      {
        name = "fsx-lustre"
        host = {
          sourcePath = "/fsx"
        }
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch_jobs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "gpu-job"
      }
    }
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 86400 # 24 hours
  }

  tags = local.common_tags
}

# ==========================================
# CloudWatch and SNS for Monitoring
# ==========================================

resource "aws_cloudwatch_log_group" "batch_jobs" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = 7

  tags = local.common_tags
}

resource "aws_sns_topic" "batch_failures" {
  name = "${var.project_name}-batch-failures"

  tags = merge(local.common_tags, {
    Name = "Batch Job Failures"
  })
}

resource "aws_sns_topic_subscription" "batch_failures_email" {
  topic_arn = aws_sns_topic.batch_failures.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

resource "aws_cloudwatch_event_rule" "batch_failures" {
  name        = "${var.project_name}-batch-failures"
  description = "Capture Batch job failures"

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      status   = ["FAILED"]
      jobQueue = [aws_batch_job_queue.gpu_queue.arn]
    }
  })

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "sns" {
  rule      = aws_cloudwatch_event_rule.batch_failures.name
  target_id = "SendToSNS"
  arn       = aws_sns_topic.batch_failures.arn

  input_transformer {
    input_paths = {
      job    = "$.detail.jobName"
      status = "$.detail.status"
      reason = "$.detail.statusReason"
    }

    input_template = "\"Batch Job Failed: Job=<job>, Status=<status>, Reason=<reason>\""
  }
}

resource "aws_sns_topic_policy" "batch_failures" {
  arn = aws_sns_topic.batch_failures.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "events.amazonaws.com"
      }
      Action   = "SNS:Publish"
      Resource = aws_sns_topic.batch_failures.arn
      Condition = {
        StringEquals = {
          "aws:SourceArn" = aws_cloudwatch_event_rule.batch_failures.arn
        }
      }
    }]
  })
}

# ==========================================
# ECS Fargate: data-ui (Streamlit)
# ==========================================

resource "aws_ecr_repository" "data_ui" {
  name                 = "data-ui"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.common_tags, {
    Name = "Data UI Container Registry"
  })
}

resource "aws_ecs_cluster" "data_ui" {
  name = "${var.project_name}-data-ui"
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "data_ui" {
  name              = "/ecs/${var.project_name}-data-ui"
  retention_in_days = 7
  tags              = local.common_tags
}

# Public SG to expose 8501 (Streamlit). For ALB usage, tighten to ALB SG instead.
resource "aws_security_group" "data_ui" {
  name_prefix = "${var.project_name}-data-ui-"
  vpc_id      = local.vpc_id
  description = "Security group for data-ui service"

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = [var.data_ui_ingress_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-data-ui-sg"
  })
}

# Look up an existing Secrets Manager secret by name (do NOT store secret in TF)
data "aws_secretsmanager_secret" "data_ui_creds" {
  name = var.data_ui_secret_name
}

# Execution role for ECS task (pull image, logs, and read the secret JSON keys)
resource "aws_iam_role" "data_ui_execution" {
  name = "${var.project_name}-data-ui-exec-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ecs-tasks.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "data_ui_exec_base" {
  role       = aws_iam_role.data_ui_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "data_ui_exec_secrets" {
  name = "${var.project_name}-data-ui-exec-secrets"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = [
        "secretsmanager:GetSecretValue",
        "kms:Decrypt"
      ],
      Resource = [
        data.aws_secretsmanager_secret.data_ui_creds.arn,
        "arn:aws:kms:${var.aws_region}:${local.account_id}:key/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "data_ui_exec_secrets_attach" {
  role       = aws_iam_role.data_ui_execution.name
  policy_arn = aws_iam_policy.data_ui_exec_secrets.arn
}

resource "aws_ecs_task_definition" "data_ui" {
  family                   = "${var.project_name}-data-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.data_ui_cpu
  memory                   = var.data_ui_memory
  execution_role_arn       = aws_iam_role.data_ui_execution.arn
  # No task role needed if using user keys; add later if migrating to IAM role auth.

  container_definitions = jsonencode([
    {
      name      = "data-ui",
      image     = "${aws_ecr_repository.data_ui.repository_url}:latest",
      essential = true,
      portMappings = [{
        containerPort = 8501,
        protocol      = "tcp"
      }],
      environment = [
        { name = "AWS_REGION", value = var.aws_region },
        { name = "ACCOUNT_ID", value = local.account_id },
        { name = "ECR_REPOSITORY", value = "${aws_ecr_repository.gpu_jobs.repository_url}" },
        { name = "RAW_BUCKET", value = aws_s3_bucket.raw_data.bucket },
        { name = "FINISHED_BUCKET", value = aws_s3_bucket.finished_data.bucket },
        { name = "CHECKPOINTS_BUCKET", value = aws_s3_bucket.checkpoints.bucket },
        { name = "STREAMLIT_SERVER_ENABLE_CORS", value = "false" },
        { name = "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", value = "true" }
      ],
      secrets = [
        {
          name      = "AWS_ACCESS_KEY_ID",
          valueFrom = "${data.aws_secretsmanager_secret.data_ui_creds.arn}:AWS_ACCESS_KEY::"
        },
        {
          name      = "AWS_SECRET_ACCESS_KEY",
          valueFrom = "${data.aws_secretsmanager_secret.data_ui_creds.arn}:AWS_SECRET_KEY::"
        }
      ],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.data_ui.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "data-ui"
        }
      }
    }
  ])

  tags = local.common_tags
}

resource "aws_ecs_service" "data_ui" {
  name                               = "${var.project_name}-data-ui"
  cluster                            = aws_ecs_cluster.data_ui.id
  task_definition                    = aws_ecs_task_definition.data_ui.arn
  desired_count                      = var.data_ui_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = module.vpc.public_subnets
    security_groups  = [aws_security_group.data_ui.id]
    assign_public_ip = true
  }

  tags = local.common_tags
}