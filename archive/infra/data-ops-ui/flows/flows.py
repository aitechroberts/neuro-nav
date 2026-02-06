# flows.py
# Prefect v3 orchestration for GPU Batch jobs with optional S3 upload.
# - Push-based: back deployments with a push work pool (no polling workers).
# - Config via Prefect Variables (lowercase/underscores), with env fallbacks:
#     raw_bucket, finished_bucket, batch_job_queue, batch_job_definition,
#     checkpoints_bucket (optional), evaluations_bucket (optional)
# - AWS credentials loaded from Prefect Block "neuro-nav-aws-creds"
#
# UI should call create_flow_run_from_deployment(...) with parameters to run.

import os
import time
from typing import Optional
from base64 import b64decode

import boto3
from botocore.config import Config
from prefect import flow, task, get_run_logger
from prefect.variables import Variable

# AWS credentials from Prefect Block (for managed runner)
# Falls back to environment/IAM if block not found (for local dev)
try:
    from prefect_aws import AwsCredentials
    _aws_creds_block = AwsCredentials.load("neuro-nav-aws-creds")
    _boto_session = _aws_creds_block.get_boto3_session()
    print("[flows] Loaded AWS credentials from Prefect Block 'neuro-nav-aws-creds'")
except Exception as e:
    # Fallback to default credentials (env vars, IAM role, etc.)
    _boto_session = boto3.Session()
    print(f"[flows] Using default AWS credentials (Block not found: {e})")


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
CHECKPOINTS_BUCKET = Variable.get(
    "checkpoints_bucket",
    default=os.getenv("CHECKPOINTS_BUCKET", ""),
)
EVALUATIONS_BUCKET = Variable.get(
    "evaluations_bucket",
    default=os.getenv("EVALUATIONS_BUCKET", os.getenv("DATASETS_BUCKET", "")),
)

# Build boto3 clients from session (uses Block credentials if loaded)
boto_cfg = Config(region_name=AWS_REGION, retries={"max_attempts": 10, "mode": "adaptive"})
s3 = _boto_session.client("s3", config=boto_cfg, region_name=AWS_REGION)
batch = _boto_session.client("batch", config=boto_cfg, region_name=AWS_REGION)


# -------------------------
# Dynamic Job Definition
# -------------------------

def _register_job_definition_with_image(base_job_def: str, new_image: str) -> str:
    """
    Registers a new job definition revision based on an existing one but with a different image.
    
    Args:
        base_job_def: ARN or name of the base job definition to clone settings from
        new_image: The new container image URI (e.g., 'account.dkr.ecr.region.amazonaws.com/repo:tag')
    
    Returns:
        The ARN of the newly registered job definition
    """
    # Describe the base job definition to get its settings
    # If it's an ARN, extract just the name for describe
    if base_job_def.startswith("arn:"):
        # ARN format: arn:aws:batch:region:account:job-definition/name:revision
        job_def_name = base_job_def.split("/")[-1].split(":")[0]
    else:
        job_def_name = base_job_def.split(":")[0]  # Remove revision if present
    
    describe_resp = batch.describe_job_definitions(
        jobDefinitionName=job_def_name,
        status="ACTIVE",
    )
    
    if not describe_resp.get("jobDefinitions"):
        raise ValueError(f"No active job definition found with name: {job_def_name}")
    
    # Get the latest revision
    base_def = sorted(
        describe_resp["jobDefinitions"],
        key=lambda d: d.get("revision", 0),
        reverse=True
    )[0]
    
    # Build the new job definition with the new image
    container_props = base_def.get("containerProperties", {}).copy()
    container_props["image"] = new_image
    
    # Generate a unique name for this dynamic job definition
    timestamp = int(time.time())
    dynamic_name = f"{job_def_name}-dynamic-{timestamp}"
    
    # Register the new job definition
    register_resp = batch.register_job_definition(
        jobDefinitionName=dynamic_name,
        type=base_def.get("type", "container"),
        platformCapabilities=base_def.get("platformCapabilities", ["EC2"]),
        containerProperties=container_props,
        retryStrategy=base_def.get("retryStrategy", {"attempts": 1}),
        timeout=base_def.get("timeout", {"attemptDurationSeconds": 86400}),
        tags=base_def.get("tags", {}),
    )
    
    new_arn = register_resp["jobDefinitionArn"]
    print(f"[info] Registered dynamic job definition: {new_arn} with image: {new_image}")
    return new_arn


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
    output_folder: str,
) -> str:
    """
    Submit an AWS Batch job. If image is provided, creates a dynamic job definition
    with that image before submitting.
    
    Args:
        job_def_arn: Base job definition ARN (used as template if image override requested)
        image: Optional image URI to use instead of the job definition's default
        output_folder: Folder path (prefix) in the finished bucket where all 
                       outputs (point cloud, semantic snapshot, etc.) will be synced.
    """
    # If image override requested, register a dynamic job definition
    effective_job_def = job_def_arn
    if image:
        print(f"[info] Image override requested: {image}")
        effective_job_def = _register_job_definition_with_image(job_def_arn, image)
    
    # Construct URIs expected by entrypoint_batch_vlm.sh
    s3_input_uri = f"s3://{RAW_BUCKET}/{input_key}"
    
    # Ensure output_folder doesn't have leading/trailing slashes, then add trailing slash
    output_folder_clean = output_folder.strip("/") if output_folder else "default"
    
    # entrypoint does: aws s3 sync OUTPUT_ROOT S3_OUTPUT_URI
    # So S3_OUTPUT_URI should be a folder (prefix) with trailing slash
    s3_output_uri = f"s3://{FINISHED_BUCKET}/{output_folder_clean}/"
    
    # Extract SCENE_ID from input_key (e.g., 'replica/room0.zip' -> 'room0')
    scene_id = os.path.splitext(os.path.basename(input_key))[0]

    # EC2-based Batch jobs support: vcpus, memory, command, instanceType, environment, resourceRequirements
    container_overrides = {
        "environment": [
            {"name": "S3_INPUT_URI", "value": s3_input_uri},
            {"name": "S3_OUTPUT_URI", "value": s3_output_uri},
            {"name": "SCENE_ID", "value": scene_id},
            {"name": "RAW_BUCKET", "value": RAW_BUCKET},
            {"name": "INPUT_KEY", "value": input_key},
            {"name": "OUTPUT_BUCKET", "value": FINISHED_BUCKET},
            {"name": "OUTPUT_FOLDER", "value": output_folder_clean},
        ],
        "resourceRequirements": [{"type": "GPU", "value": "1"}],
    }

    resp = batch.submit_job(
        jobName=f"gpu-pipeline-{os.getpid()}",
        jobQueue=job_queue,
        jobDefinition=effective_job_def,
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
    output_folder: str,
    override_image: bool = False,
) -> Optional[str]:
    logger = get_run_logger()
    if not run_batch:
        logger.info("run_batch=False; skipping Batch stage.")
        return None

    image = f"{ecr_repo}:{ecr_tag}" if override_image and ecr_repo and ecr_tag else None
    job_id = _submit_batch_job(job_queue, job_def_arn, image, input_key, output_folder)
    logger.info(f"Submitted Batch job: {job_id}")
    return job_id


# -------------------------
# Flows
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
    # Output folder (all outputs synced here)
    output_folder: str = "default",
    # Control whether to actually submit batch
    run_batch: bool = True,
    # Optional: also trigger an evaluation flow after Batch submission
    run_evaluation_after: bool = False,
    evaluations_bucket: Optional[str] = None,
    # Legacy parameter name (mapped to output_folder for backward compat)
    output_name: Optional[str] = None,
):
    """
    Unified pipeline:
      - 'upload' mode: uploads base64 data to s3://<raw_bucket>/<upload_key>, then (optionally) submits AWS Batch job.
      - 'existing' mode: skips upload and submits AWS Batch job against s3://<raw_bucket>/<existing_key>.
    
    All outputs (point cloud, semantic snapshot, etc.) are synced to:
      s3://<finished_bucket>/<output_folder>/
    
    Dynamic Image Support:
      If override_image=True and ecr_repo/ecr_tag are provided, a new job definition
      will be registered with that image before submitting the job.
    """

    logger = get_run_logger()
    logger.info("Starting gpu-pipeline")

    # Handle legacy output_name parameter
    resolved_output_folder = output_name if output_name else output_folder

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
        resolved_output_folder,
        override_image=override_image,
    )

    # Clean output folder for URI construction
    output_folder_clean = resolved_output_folder.strip("/") if resolved_output_folder else "default"

    # Optionally kick off an evaluation flow after submitting the Batch job.
    evaluation_result: Optional[dict] = None
    if run_batch and run_evaluation_after and job_id:
        eval_output_folder = f"evaluation-{output_folder_clean}"
        evaluation_result = evaluate_results(
            input_folder=output_folder_clean,
            finished_bucket=FINISHED_BUCKET,
            evaluations_bucket=evaluations_bucket,
            batch_job_queue=resolved_queue,
            evaluation_job_definition_arn=resolved_job_def,
            ecr_repo=ecr_repo,
            ecr_tag=ecr_tag,
            override_image=override_image,
            output_folder=eval_output_folder,
        )

    result = {
        "mode": data_mode,
        "input_key": input_key,
        "raw_bucket": target_bucket,
        "finished_bucket": FINISHED_BUCKET,
        "batch_job_queue": resolved_queue,
        "batch_job_definition": resolved_job_def,
        "batch_job_id": job_id,
        "output_uri": f"s3://{FINISHED_BUCKET}/{output_folder_clean}/",
        "evaluations_bucket": evaluation_result.get("evaluations_bucket") if evaluation_result else None,
        "evaluation_job_id": evaluation_result.get("evaluation_job_id") if evaluation_result else None,
        "evaluation_output_uri": evaluation_result.get("evaluation_output_uri") if evaluation_result else None,
    }
    logger.info(result)
    return result


@flow
def evaluate_results(
    input_folder: str,
    finished_bucket: Optional[str] = None,
    evaluations_bucket: Optional[str] = None,
    batch_job_queue: Optional[str] = None,
    evaluation_job_definition_arn: Optional[str] = None,
    ecr_repo: str = "",
    ecr_tag: str = "",
    override_image: bool = False,
    output_folder: str = "evaluation",
    # Legacy parameter (mapped to input_folder)
    input_key: Optional[str] = None,
    output_name: Optional[str] = None,
):
    """
    Evaluation-only flow:
      - Reads from s3://<finished_bucket>/<input_folder>/
      - Submits an evaluation AWS Batch job (using gpu-jobs image)
      - Evaluation job writes to s3://<evaluations_bucket>/<output_folder>/
    
    Dynamic Image Support:
      If override_image=True and ecr_repo/ecr_tag are provided, a new job definition
      will be registered with that image before submitting the job.
    """
    logger = get_run_logger()
    logger.info("Starting evaluate-results flow")

    # Handle legacy parameters
    resolved_input_folder = input_key if input_key else input_folder
    resolved_output_folder = output_name if output_name else output_folder

    resolved_finished_bucket = finished_bucket or FINISHED_BUCKET
    if not resolved_finished_bucket:
        raise ValueError(
            "No finished_bucket provided and Prefect Variable 'finished_bucket' is not set."
        )

    resolved_evaluations_bucket = evaluations_bucket or EVALUATIONS_BUCKET
    if not resolved_evaluations_bucket:
        raise ValueError(
            "No evaluations_bucket provided and Prefect Variable 'evaluations_bucket' is not set."
        )

    resolved_queue = batch_job_queue or BATCH_JOB_QUEUE
    if not resolved_queue:
        raise ValueError(
            "No batch_job_queue provided and Prefect Variable 'batch_job_queue' is not set."
        )

    resolved_job_def = evaluation_job_definition_arn or BATCH_JOB_DEF
    if not resolved_job_def:
        raise ValueError(
            "No evaluation_job_definition_arn provided and Prefect Variable "
            "'batch_job_definition' is not set."
        )

    # Clean folder paths
    input_folder_clean = resolved_input_folder.strip("/") if resolved_input_folder else "default"
    output_folder_clean = resolved_output_folder.strip("/") if resolved_output_folder else "evaluation"

    # If image override requested, register a dynamic job definition
    image = f"{ecr_repo}:{ecr_tag}" if override_image and ecr_repo and ecr_tag else None
    effective_job_def = resolved_job_def
    if image:
        logger.info(f"Image override requested: {image}")
        effective_job_def = _register_job_definition_with_image(resolved_job_def, image)

    # Submit an evaluation job. The evaluation container code is responsible for:
    # - Reading from EVAL_INPUT_BUCKET/EVAL_INPUT_FOLDER
    # - Writing to EVAL_OUTPUT_BUCKET/EVAL_OUTPUT_FOLDER
    container_overrides = {
        "environment": [
            {"name": "EVAL_INPUT_BUCKET", "value": resolved_finished_bucket},
            {"name": "EVAL_INPUT_FOLDER", "value": input_folder_clean},
            {"name": "EVAL_OUTPUT_BUCKET", "value": resolved_evaluations_bucket},
            {"name": "EVAL_OUTPUT_FOLDER", "value": output_folder_clean},
        ],
        "resourceRequirements": [{"type": "GPU", "value": "1"}],
    }

    resp = batch.submit_job(
        jobName=f"evaluate-results-{os.getpid()}",
        jobQueue=resolved_queue,
        jobDefinition=effective_job_def,
        containerOverrides=container_overrides,
        propagateTags=True,
    )
    evaluation_job_id = resp["jobId"]
    logger.info(f"Submitted evaluation Batch job: {evaluation_job_id}")

    evaluation_output_uri = f"s3://{resolved_evaluations_bucket}/{output_folder_clean}/"
    result = {
        "input_finished_uri": f"s3://{resolved_finished_bucket}/{input_folder_clean}/",
        "evaluations_bucket": resolved_evaluations_bucket,
        "evaluation_job_id": evaluation_job_id,
        "evaluation_output_uri": evaluation_output_uri,
    }
    logger.info(result)
    return result
