variable "iam_user_names" {
  description = "List of IAM usernames to be managed and added to the group."
  type        = list(string)
}

variable "iam_group_name" {
  description = "IAM group name that users will be assigned to."
  type        = string
}


