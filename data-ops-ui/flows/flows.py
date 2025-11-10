# flows.py
# Prefect v3 orchestration for GPU Batch jobs with optional S3 upload.
# - Push-based: back deployments with a push work pool (no polling workers).
# - Config via Prefect Variables (lowercase/underscores), with env fallbacks:
#     raw_bucket, finished_bucket, batch_job_queue, batch_job_definition,
#     fsx_path (optional), checkpoints_bucket (optional)
#
# UI should call create_flow_run_from_deployment(...) with parameters to run.

import os
from typing import Optional
from base64 import b64decode

import boto3
from botocore.config import Config
from prefect import flow, task, get_run_logger
from prefect.variables import Variable


# -------------------------
# Clients / configuration
# -------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Prefer Prefect Variables; allow env fallback so local dev still works.
RAW_BUCKET = Variable.get("raw_bucket", default=os.getenv("RAW_BUCKET", ""))
FINISHED_BUCKET = Variable.get(
    "finished_bucket",
    default=os.getenv("FINISHED_BUCKET", RAW_BUCKET),
)
BATCH_JOB_QUEUE = Variable.get(
    "batch_job_queue",
    default=os.getenv("BATCH_JOB_QUEUE", ""),
)
BATCH_JOB_DEF = Variable.get(
    "batch_job_definition",
    default=os.getenv("BATCH_JOB_DEFINITION", ""),
)

# Optional variables you might read in your job or expose as env to batch:
FSX_PATH = Variable.get("fsx_path", default=os.getenv("FSX_PATH", "/fsx/checkpoints"))
CHECKPOINTS_BUCKET = Variable.get(
    "checkpoints_bucket",
    default=os.getenv("CHECKPOINTS_BUCKET", ""),
)

boto_cfg = Config(region_name=AWS_REGION, retries={"max_attempts": 10, "mode": "adaptive"})
s3 = boto3.client("s3", config=boto_cfg)
batch = boto3.client("batch", config=boto_cfg)


# -------------------------
# Tasks
# -------------------------

@task
def upload_content_to_s3(bucket: str, key: str, content_b64: str, content_type: Optional[str] = None) -> str:
    """
    Decodes base64 content and uploads it to s3://bucket/key.
    Returns the key.
    """
    data = b64decode(content_b64)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3.put_object(Bucket=bucket, Key=key, Body=data, **extra)
    return key


def _submit_batch_job(
    job_queue: str,
    job_def_arn: str,
    image: Optional[str],
    input_key: str,
    output_name: str,
    ckpt_dir: str,
) -> str:
    """
    Submit an AWS Batch job. Optionally override the container image.
    Passes RAW_BUCKET/INPUT_KEY/OUTPUT_BUCKET/OUTPUT_NAME/CKPT_DIR via env.
    """
    container_overrides = {
        "name": "main",
        "environment": [
            {"name": "RAW_BUCKET", "value": RAW_BUCKET},
            {"name": "INPUT_KEY", "value": input_key},
            {"name": "OUTPUT_BUCKET", "value": FINISHED_BUCKET},
            {"name": "OUTPUT_NAME", "value": output_name},
            {"name": "CKPT_DIR", "value": ckpt_dir},
        ],
        # You can remove this if your job definition already sets GPU requirements.
        "resourceRequirements": [{"type": "GPU", "value": "1"}],
    }

    # Only override the image if explicitly requested
    if image:
        container_overrides["image"] = image

    resp = batch.submit_job(
        jobName=f"gpu-pipeline-{os.getpid()}",
        jobQueue=job_queue,
        jobDefinition=job_def_arn,
        containerOverrides=container_overrides,
        propagateTags=True,
    )
    return resp["jobId"]


@task
def submit_batch_if_requested(
    run_batch: bool,
    job_queue: str,
    job_def_arn: str,
    ecr_repo: str,
    ecr_tag: str,
    input_key: str,
    output_name: str,
    ckpt_dir: str,
    override_image: bool = False,
) -> Optional[str]:
    logger = get_run_logger()
    if not run_batch:
        logger.info("run_batch=False; skipping Batch stage.")
        return None

    image = f"{ecr_repo}:{ecr_tag}" if override_image and ecr_repo and ecr_tag else None
    job_id = _submit_batch_job(job_queue, job_def_arn, image, input_key, output_name, ckpt_dir)
    logger.info(f"Submitted Batch job: {job_id}")
    return job_id


# -------------------------
# Flow
# -------------------------

@flow
def gpu_pipeline(
    # Mode: "upload" or "existing"
    data_mode: str,
    # If data_mode == "upload": provide upload_key + upload_content_b64
    upload_key: Optional[str] = None,
    upload_content_b64: Optional[str] = None,
    content_type: Optional[str] = None,
    # If data_mode == "existing": provide existing_key
    existing_key: Optional[str] = None,
    # Optional overrides (prefer Prefect Variables by default)
    raw_bucket: Optional[str] = None,
    batch_job_queue: Optional[str] = None,
    batch_job_definition_arn: Optional[str] = None,
    # Batch image override controls
    ecr_repo: str = "",
    ecr_tag: str = "",
    override_image: bool = False,
    # Output and mounts (your container reads these from env)
    output_name: str = "results.json",
    fsx_mount_path: str = "/fsx",     # informational; not used directly here
    ckpt_mount: str = "/fsx/checkpoints",
    # Control whether to actually submit batch
    run_batch: bool = True,
):
    """
    Unified pipeline:
      - 'upload' mode: uploads base64 data to s3://<raw_bucket>/<upload_key>, then (optionally) submits AWS Batch job.
      - 'existing' mode: skips upload and submits AWS Batch job against s3://<raw_bucket>/<existing_key>.
    """

    logger = get_run_logger()
    logger.info("Starting gpu-pipeline")

    # Resolve config from parameters -> Prefect Variables -> env defaults
    target_bucket = raw_bucket or RAW_BUCKET
    if not target_bucket:
        raise ValueError(
            "No raw_bucket provided and Prefect Variable 'raw_bucket' is not set. "
            "Set it via `prefect variable set raw_bucket <your-bucket>` or pass raw_bucket param."
        )

    resolved_queue = batch_job_queue or BATCH_JOB_QUEUE
    if not resolved_queue:
        raise ValueError(
            "No batch_job_queue provided and Prefect Variable 'batch_job_queue' is not set."
        )

    resolved_job_def = batch_job_definition_arn or BATCH_JOB_DEF
    if not resolved_job_def:
        raise ValueError(
            "No batch_job_definition provided and Prefect Variable 'batch_job_definition' is not set."
        )

    # Determine input key based on mode
    key_from_upload = None
    if data_mode == "upload":
        if not upload_key or not upload_content_b64:
            raise ValueError("Upload mode requires both upload_key and upload_content_b64.")
        key_from_upload = upload_content_to_s3(
            target_bucket, upload_key, upload_content_b64, content_type
        )
        logger.info(f"Uploaded s3://{target_bucket}/{upload_key}")

    elif data_mode == "existing":
        if not existing_key:
            raise ValueError("Existing mode requires existing_key.")
    else:
        raise ValueError("data_mode must be 'upload' or 'existing'.")

    input_key = key_from_upload or existing_key  # type: ignore[assignment]
    if not input_key:
        raise ValueError("Unable to resolve input_key from provided parameters.")

    # Submit Batch if requested
    job_id = submit_batch_if_requested(
        run_batch,
        resolved_queue,
        resolved_job_def,
        ecr_repo,
        ecr_tag,
        input_key,
        output_name,
        ckpt_mount,
        override_image=override_image,
    )

    result = {
        "mode": data_mode,
        "input_key": input_key,
        "raw_bucket": target_bucket,
        "finished_bucket": FINISHED_BUCKET,
        "batch_job_queue": resolved_queue,
        "batch_job_definition": resolved_job_def,
        "batch_job_id": job_id,
        "fsx_mount_path": fsx_mount_path,
        "ckpt_mount": ckpt_mount,
        "output_uri": f"s3://{FINISHED_BUCKET}/{output_name}",
    }
    logger.info(result)
    return result
