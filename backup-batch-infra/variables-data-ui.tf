# ==========================================
# ECS Fargate Data UI Variables
# ==========================================

variable "prefect_api_url" {
  description = "Prefect Cloud API URL"
  type        = string
  default     = "https://api.prefect.cloud/api/accounts/0ba00981-f3c1-47c4-ae5c-6ceb9b83a9a1/workspaces/4cc2d567-caa8-400e-80bf-95038426a9af"
}
