variable "data_ui_ingress_cidr" {
  description = "CIDR allowed to access Streamlit UI on port 8501 (use ALB SG in prod)."
  type        = string
  default     = "0.0.0.0/0"
}

variable "data_ui_secret_name" {
  description = "Secrets Manager name to store IAM user creds JSON."
  type        = string
  default     = "User-Keys"
}

variable "data_ui_cpu" {
  description = "Fargate CPU units for data-ui"
  type        = string
  default     = "1024"
}

variable "data_ui_memory" {
  description = "Fargate memory (MiB) for data-ui"
  type        = string
  default     = "6144"
}

variable "data_ui_desired_count" {
  description = "Desired task count for data-ui service"
  type        = number
  default     = 1
}


