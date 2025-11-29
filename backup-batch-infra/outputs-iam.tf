output "iam_group_name" {
  description = "IAM group name"
  value       = aws_iam_group.team.name
}

output "iam_group_arn" {
  description = "IAM group ARN"
  value       = aws_iam_group.team.arn
}

output "iam_user_arns" {
  description = "Map of IAM usernames to their ARNs"
  value       = { for uname, u in aws_iam_user.users : uname => u.arn }
}


