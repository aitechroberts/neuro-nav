Current data-ui IP Address: 98.92.111.66
Accessed with http://98.92.111.66:8501
- NOTE, not https

### S3 Bucket Names
- 	
data-finished-585780419748-us-east-1
- data-raw-585780419748-us-east-1
- model-checkpoints-585780419748-us-east-1

## Deploy Streamlit Image instructions

### Set deployment variables
```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=585780419748
export ECR_REPO=data-ui
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export IMAGE_NAME=data-ui
export IMAGE_TAG=latest
```

### Build and push image
```bash
docker build -t "${ECR_REPO}:${IMAGE_TAG}" .
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# Log in to ECR
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"
docker push "${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
```

Then update ECS to force new deployment for newly pushed image
```bash
aws ecs update-service \
  --cluster gpu-batch-data-ui \
  --service gpu-batch-data-ui \
  --force-new-deployment
```

### Deploy Prefect Worker

```bash
prefect work-pool create managed-push --type prefect:managed
prefect variables set RAW_BUCKET=data-raw-585780419748-us-east-1
prefect variables set FINISHED_BUCKET=data-finished-585780419748-us-east-1
prefect variables set BATCH_JOB_QUEUE=gpu-batch-gpu-queue
prefect variables set BATCH_JOB_DEFINITION=gpu-batch-gpu-generic:1
prefect variables set FSX_PATH=/fsx/checkpoints
```