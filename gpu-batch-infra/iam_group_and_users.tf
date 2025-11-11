resource "aws_iam_group" "team" {
  name = var.iam_group_name
}

resource "aws_iam_group_policy_attachment" "s3_full" {
  group      = aws_iam_group.team.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_group_policy_attachment" "ecr_public_full" {
  group      = aws_iam_group.team.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicFullAccess"
}

resource "aws_iam_user" "users" {
  for_each = toset(var.iam_user_names)
  name     = each.value
  tags     = local.common_tags
}

# Manages the exact membership of the group to match var.iam_user_names
resource "aws_iam_group_membership" "team_members" {
  name  = "${var.iam_group_name}-membership"
  group = aws_iam_group.team.name
  users = var.iam_user_names
}


