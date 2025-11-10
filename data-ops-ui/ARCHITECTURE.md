## Neuro-Nav Data Ops - Architecture and Operations

This document describes how the Neuro‑Nav data ops stack is wired, how data flows end‑to‑end, and how to operate and extend it. It is focused on the Streamlit “data‑ops UI”, AWS infrastructure (ECS Fargate, ECR, S3, IAM), and Prefect orchestration for GPU Batch jobs.

### High-level overview
- Streamlit UI (`data-ops-ui/app.py`) runs on ECS Fargate as the “Data UI” service. It lets users:
  - Upload a scene archive to the raw S3 bucket (via Prefect) or pick an existing object
  - Select an image tag from the `gpu-jobs` ECR repo
  - Optionally trigger an AWS Batch GPU job for processing (via Prefect)
- Prefect Cloud orchestrates the flow (`data-ops-ui/flows/flows.py`) with push-based deployments defined in `data-ops-ui/prefect.yaml`.
- AWS resources are provisioned by Terraform in `gpu-batch-infra/`:
  - S3 buckets: raw, finished, checkpoints
  - ECR repos: `gpu-jobs` (GPU workers) and `data-ui` (Streamlit UI)
  - Batch (GPU spot compute env, queue, job definition)
  - ECS Fargate cluster/service for the Data UI (public IP for quick access)
  - IAM policies/roles and an optional cross‑account role for collaborators


## Components

### 1) Streamlit UI (ECS Fargate)
- Code: `neuro-nav/data-ops-ui/app.py`
- Container image: `data-ui` ECR repository
- Terraform: `aws_ecs_task_definition.data_ui` + `aws_ecs_service.data_ui`
  - Port 8501 exposed; public IP enabled for quick access (consider ALB later)
  - Reads AWS credentials at runtime from a Secrets Manager secret (JSON)
    - Secret keys in our secret: `AWS_ACCESS_KEY` and `AWS_SECRET_KEY`
    - In task definition, they are mapped to env vars `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- The UI retrieves ECR tags, lists S3 objects, and triggers Prefect deployments.

### 2) Prefect (Cloud)
- Flow code: `neuro-nav/data-ops-ui/flows/flows.py`
- Deployments: `neuro-nav/data-ops-ui/prefect.yaml` defines three entry points:
  - `upload-only`: upload to S3, no Batch
  - `run-existing`: run Batch against an existing S3 object
  - `upload-and-run`: upload, then run Batch
- Work pool: push‑based (managed), no persistent agent required. The Streamlit app uses `run_deployment(...)` to submit runs.
- Configuration:
  - Flow resolves bucket/queue/job_def via parameters first, then Prefect Variables (lowercase), then env defaults.
  - Recommended Variables (Prefect Cloud): `raw_bucket`, `finished_bucket`, `batch_job_queue`, `batch_job_definition`, `fsx_path`, `checkpoints_bucket`.

### 3) AWS Batch (GPU)
- Terraform resources provision compute environment, queue, and job definition.
- Flow submits a job with container overrides for input/output, checkpoint dir, and (optionally) overrides image from `gpu-jobs` with the selected tag.

### 4) S3 buckets (data layout)
- Raw bucket: `data-raw-<account>-<region>`
- Finished bucket: `data-finished-<account>-<region>`
- Checkpoints bucket: `model-checkpoints-<account>-<region>`
- Recommended: upload one archive (`.tar.gz` or `.zip`) per scene, with a small manifest (see README for bundle schema).


## Data flow (end‑to‑end)
1. User opens the Data UI (public IP: `http://<public-ip>:8501`). Sidebar shows resolved config (buckets, ECR repo, Prefect deployment).
2. Choose data:
   - Upload new data: UI base64‑encodes the file and submits a Prefect run with `data_mode="upload"`. The flow uploads to `s3://<raw_bucket>/<upload_key>`.
   - Select existing data: UI lists keys in the raw bucket and submits with `data_mode="existing"`.
3. Optionally choose a GPU image tag from `gpu-jobs` and toggle whether to override the Batch job’s image.
4. UI calls `run_deployment(...)` (non‑blocking). Prefect creates a flow run.
5. Flow uploads (if requested), then submits an AWS Batch job (if `run_batch=true`), passing the input S3 key and output name.
6. Outputs are written to the finished bucket; logs/metrics are visible in Prefect and CloudWatch.


## Terraform (infra) – files to know
- `gpu-batch-infra/main.tf`
  - VPC, FSx Lustre, S3, ECR, Batch, ECS cluster/service for Data UI, Secrets Manager lookup
  - ECR cross‑account policy `aws_ecr_repository_policy.gpu_jobs_cross_account`
  - Cross‑account role `aws_iam_role.data_ops_contributor`
- `gpu-batch-infra/variables-data-ui.tf`
  - Public ingress CIDR (for dev), Fargate CPU/Mem (1 vCPU, 6 GB), desired count, secret name
- `gpu-batch-infra/terraform.tfvars`
  - Region, environment, team account IDs (for cross‑account ECR), external ID for role assumption, batch/FSx sizing
- `gpu-batch-infra/outputs.tf`
  - Prints useful ARNs/URLs, including cross‑account role ARN and external ID instructions


## Operating the Data UI

### Build and push the Data UI image
```bash
AWS_REGION=us-east-1
ACCOUNT_ID=<your-account-id>
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t data-ui:latest -f neuro-nav/data-ops-ui/Dockerfile neuro-nav/data-ops-ui
docker tag data-ui:latest "${REGISTRY}/data-ui:latest"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
docker push "${REGISTRY}/data-ui:latest"
```
Force a new ECS deployment so tasks pull the new image:
```bash
aws ecs update-service \
  --cluster gpu-batch-data-ui \
  --service gpu-batch-data-ui \
  --force-new-deployment
```

### Updating secrets (UI AWS creds)
Store a JSON secret in AWS Secrets Manager (name default: `User-Keys`):
```json
{ "AWS_ACCESS_KEY": "AKIA...", "AWS_SECRET_KEY": "..." }
```
The task definition maps those keys to the canonical envs (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). No changes required in the app.

### Prefect deployments
```bash
cd neuro-nav/data-ops-ui
prefect deploy -n upload-only
prefect deploy -n run-existing
prefect deploy -n upload-and-run
```
Set variables (lowercase) in Prefect Cloud to avoid passing them from the UI every time:
```bash
prefect variables set raw_bucket data-raw-<acct>-us-east-1
prefect variables set finished_bucket data-finished-<acct>-us-east-1
prefect variables set batch_job_queue gpu-batch-gpu-queue
prefect variables set batch_job_definition arn:aws:batch:us-east-1:<acct>:job-definition/<name>:<rev>
prefect variables set fsx_path /fsx/checkpoints
prefect variables set checkpoints_bucket model-checkpoints-<acct>-us-east-1
```


## External contributors: add a new account and push GPU images
You have two collaboration paths; choose one.

### A) Cross‑account ECR policy (simple push/pull)
Terraform attaches a repository policy on `gpu-jobs` that grants push/pull to root principals of accounts listed in `team_accounts`:
1. Add the external account ID to `gpu-batch-infra/terraform.tfvars` under `team_accounts`, then `terraform apply`.
2. External user (in their account) runs:
```bash
AWS_REGION=us-east-1
ACCOUNT_ID_OF_YOUR_ACCOUNT=<our-account-id>
REGISTRY="${ACCOUNT_ID_OF_YOUR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

docker build -t gpu-jobs:<tag> <path-to-Dockerfile>
docker tag gpu-jobs:<tag> "${REGISTRY}/gpu-jobs:<tag>"
docker push "${REGISTRY}/gpu-jobs:<tag>"
```
Notes:
- Their principal must have ECR permissions in their own account sufficient to obtain an auth token and perform cross‑account push to our repo (policy allows the principal; auth is per user/role).
- Use a unique `<tag>`; the Data UI lists tags from `gpu-jobs` for selection.

### B) Assume a cross‑account role (tighter control)
Terraform creates an assumable role `DataOpsContributor` with a required `external_id`:
1. After `terraform apply`, share the outputs:
   - `cross_account_role_arn`
   - `external_id`
2. External user assumes the role and pushes as that role:
```bash
ROLE_ARN=<provided-arn>
EXTERNAL_ID=<provided-external-id>
CREDS=$(aws sts assume-role --role-arn "$ROLE_ARN" --role-session-name contributor --external-id "$EXTERNAL_ID")
export AWS_ACCESS_KEY_ID=$(echo $CREDS | jq -r .Credentials.AccessKeyId)
export AWS_SECRET_ACCESS_KEY=$(echo $CREDS | jq -r .Credentials.SecretAccessKey)
export AWS_SESSION_TOKEN=$(echo $CREDS | jq -r .Credentials.SessionToken)

AWS_REGION=us-east-1
ACCOUNT_ID_OF_YOUR_ACCOUNT=<our-account-id>
REGISTRY="${ACCOUNT_ID_OF_YOUR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
docker build -t gpu-jobs:<tag> <path-to-Dockerfile>
docker tag gpu-jobs:<tag> "${REGISTRY}/gpu-jobs:<tag>"
docker push "${REGISTRY}/gpu-jobs:<tag>"
```
Role permissions include ECR push/pull and read/write to specific S3 buckets. Use this path if you don’t want to manage per‑user policies in other accounts.


## Using the system

### Upload new data and run
1. Visit the Data UI (public IP or ALB).
2. Select “Upload new data”, choose your `.tar.gz` or `.zip`, enter destination key prefix (e.g., `replica/office0`).
3. Toggle “Run GPU batch job”, choose an ECR tag from `gpu-jobs`, and click Submit.
4. Prefect flow uploads to `s3://data-raw-.../<prefix>/<filename>` and submits an AWS Batch job.

### Run against existing data
1. Select “Select existing data”, filter by prefix, and choose the existing key.
2. Toggle Batch on/off, pick a tag (if overriding), submit.


## Updating and extending

### Update the UI
1. Edit `neuro-nav/data-ops-ui/app.py`.
2. Build and push the `data-ui:latest` image (see commands above).
3. Force new ECS deployment:
```bash
aws ecs update-service --cluster gpu-batch-data-ui --service gpu-batch-data-ui --force-new-deployment
```

### Add a new Prefect deployment
1. Edit `neuro-nav/data-ops-ui/prefect.yaml` as needed.
2. `prefect deploy -n <name>`
3. Update `PREFECT_DEPLOYMENT_PATH` in the UI env if the name/path changes.

### Change Batch image or resources
- Build/push a new tag to `gpu-jobs`, then select it in the UI when submitting.
- To pin the Batch job definition permanently, update Terraform’s job definition or pass `override_image=true` with `ecr_repo` and `ecr_tag`.


## Troubleshooting
- UI shows “ECR list failed: Unable to locate credentials”
  - Ensure ECS task has access to Secrets Manager secret and the secret JSON keys are `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` (mapped to canonical envs).
- UI deployed but no public URL
  - Terraform sets `assign_public_ip = true`. Confirm the service is running and the security group allows inbound 8501 from your IP/CIDR.
- Prefect runs stuck in Pending
  - For managed/push deployments, ensure Prefect Cloud shows runs being created and not blocked by missing variables. For worker‑based pools, start a worker.
- Batch job fails immediately
  - Verify `batch_job_queue` and `batch_job_definition` are correct, your image tag exists, and S3 keys/buckets are accessible.


## Notes and recommendations
- Prefer unique image tags or digests for repeatable runs; don’t rely solely on `:latest`.
- For large scenes, avoid browser uploads; upload with `aws s3 cp` and use “Select existing data” in the UI.
- Keep secrets in Secrets Manager; don’t pass keys via Prefect Variables or Terraform variables/state.


