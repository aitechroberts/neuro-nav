# app.py
import os
import base64
import json
from typing import List, Optional

import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Prefect (push-based trigger from the UI)
from prefect.deployments import run_deployment  # returns immediately with timeout=0  (docs)
# https://reference.prefect.io/prefect/deployments/  (run_deployment; timeout=0 to return immediately)

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Data & GPU Jobs", page_icon="🧩", layout="wide")
st.title("Data & GPU Jobs")

# -----------------------------
# AWS + environment resolution
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Resolve account (safe fallback if no creds)
try:
    _tmp_sts = boto3.client("sts", region_name=AWS_REGION)
    ACCOUNT_ID = _tmp_sts.get_caller_identity()["Account"]
except Exception:
    ACCOUNT_ID = os.getenv("ACCOUNT_ID", "585780419748")

RAW_BUCKET = os.getenv("RAW_BUCKET") or f"data-raw-{ACCOUNT_ID}-{AWS_REGION}"
FINISHED_BUCKET = os.getenv("FINISHED_BUCKET") or f"data-finished-{ACCOUNT_ID}-{AWS_REGION}"
CHECKPOINTS_BUCKET = os.getenv("CHECKPOINTS_BUCKET") or f"model-checkpoints-{ACCOUNT_ID}-{AWS_REGION}"

ECR_REPOSITORY = os.getenv("ECR_REPOSITORY") or f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/gpu-jobs"
ECR_REPO_NAME = ECR_REPOSITORY.split("/")[-1]

BATCH_JOB_QUEUE = os.getenv("BATCH_JOB_QUEUE", "gpu-batch-gpu-queue")
BATCH_JOB_DEFINITION_ARN = os.getenv("BATCH_JOB_DEFINITION_ARN", "")  # if you want to override

FSX_MOUNT_PATH = os.getenv("FSX_MOUNT_PATH", "/fsx")
CKPT_MOUNT = os.getenv("CKPT_MOUNT", "/fsx/checkpoints")

# IMPORTANT: This must match your Prefect v3 deployment "flow/deployment" path
PREFECT_DEPLOYMENT_PATH = os.getenv("PREFECT_DEPLOYMENT_PATH", "gpu_pipeline/gpu_pipeline")
# Prefect v3: deployments are server-side objects you can trigger programmatically. (docs)
# https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments

# -----------------------------
# Cached AWS clients & helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def _boto_cfg() -> Config:
    return Config(region_name=AWS_REGION, retries={"max_attempts": 10, "mode": "adaptive"})

@st.cache_resource(show_spinner=False)
def s3_client():
    return boto3.client("s3", config=_boto_cfg())

@st.cache_resource(show_spinner=False)
def ecr_client():
    return boto3.client("ecr", config=_boto_cfg())

@st.cache_resource(show_spinner=False)
def batch_client():
    return boto3.client("batch", config=_boto_cfg())

@st.cache_data(show_spinner=False, ttl=60)  # list cache to keep UI snappy
def list_s3_objects(bucket: str, prefix: str = "") -> List[str]:
    s3 = s3_client()
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys

@st.cache_data(show_spinner=False, ttl=60)
def list_ecr_image_tags(repo_name: str) -> List[str]:
    ecr = ecr_client()
    details = []
    for page in ecr.get_paginator("describe_images").paginate(repositoryName=repo_name):
        details.extend(page.get("imageDetails", []))
    details.sort(key=lambda d: d.get("imagePushedAt", 0), reverse=True)
    ordered = []
    for d in details:
        for t in d.get("imageTags", []) or []:
            if t not in ordered:
                ordered.append(t)
    return ordered

# -----------------------------
# Sidebar (read-only config)
# -----------------------------
with st.sidebar:
    st.caption("Fargate UI • Batch on G5 Spot • FSx for Lustre • Prefect v3")
    st.text_input("AWS Region", value=AWS_REGION, disabled=True)
    st.text_input("Raw Bucket", value=RAW_BUCKET or "<not set>", disabled=True)
    st.text_input("ECR Repo", value=ECR_REPOSITORY, disabled=True)
    st.text_input("Batch Queue", value=BATCH_JOB_QUEUE, disabled=True)
    st.text_input("Job Def ARN (override)", value=(BATCH_JOB_DEFINITION_ARN or "<not set>"), disabled=True)
    st.text_input("Prefect Deployment", value=PREFECT_DEPLOYMENT_PATH, disabled=True)

# -----------------------------
# Step 1: choose data
# -----------------------------
st.subheader("1) Choose data source")
data_mode_h = st.radio("Data selection mode", ["Upload new data", "Select existing data"], index=0)

upload_key: Optional[str] = None
existing_key: Optional[str] = None
uploaded_bytes: Optional[bytes] = None
file_mime: Optional[str] = None

if data_mode_h == "Upload new data":
    up = st.file_uploader("Upload input file (e.g., Replica ZIP)")
    upload_prefix = st.text_input("Destination key prefix in raw bucket", placeholder="replica/office0")
    if up is not None:
        upload_key = f"{upload_prefix.rstrip('/')}/{up.name}"
        uploaded_bytes = up.getvalue()
        file_mime = getattr(up, "type", None)
        st.info(f"Will upload to s3://{RAW_BUCKET}/{upload_key}")
else:
    prefix = st.text_input("Existing key prefix to list", value="")
    try:
        keys = list_s3_objects(RAW_BUCKET, prefix)
    except ClientError as e:
        st.error(f"S3 list failed: {e}")
        keys = []
    existing_key = st.selectbox("Choose an existing object", keys) if keys else None

st.markdown("---")

# -----------------------------
# Step 2: Batch options
# -----------------------------
st.subheader("2) Batch processing")
run_batch = st.toggle("Run GPU batch job", value=True)
selected_tag: Optional[str] = None
output_name: Optional[str] = None
override_image = st.toggle("Override Job Definition image with selected ECR tag", value=False)

if run_batch:
    try:
        tags = list_ecr_image_tags(ECR_REPO_NAME)
    except ClientError as e:
        st.error(f"ECR list failed: {e}")
        tags = []
    selected_tag = st.selectbox("Choose GPU image tag (ECR)", options=tags) if tags else None
    output_name = st.text_input("Output artifact name", value="results.json")

st.markdown("---")

# -----------------------------
# Submit → Prefect deployment
# -----------------------------
if st.button("Submit"):
    # Basic validation
    if data_mode_h == "Upload new data" and uploaded_bytes is None:
        st.error("Please choose a file to upload.")
        st.stop()
    if run_batch and (not output_name):
        st.error("Please provide an output artifact name.")
        st.stop()
    if run_batch and override_image and not selected_tag:
        st.error("You chose to override the image; please select an ECR tag.")
        st.stop()

    # Map UI → flow parameters (must match gpu_pipeline signature)
    params = {
        "data_mode": "upload" if data_mode_h == "Upload new data" else "existing",
        "upload_key": upload_key,
        "upload_content_b64": (base64.b64encode(uploaded_bytes).decode("ascii") if uploaded_bytes else None),
        "content_type": file_mime if uploaded_bytes else None,
        "existing_key": existing_key,
        # Prefer Prefect Variables inside the flow; raw_bucket here is explicit
        "raw_bucket": RAW_BUCKET,
        # Batch options
        "run_batch": run_batch,
        "ecr_repo": ECR_REPOSITORY,
        "ecr_tag": (selected_tag or ""),
        "override_image": bool(override_image),
        "batch_job_queue": BATCH_JOB_QUEUE or None,                # let flow fallback to Variable if blank
        "batch_job_definition_arn": BATCH_JOB_DEFINITION_ARN or None,
        "output_name": output_name or "results.json",
        "fsx_mount_path": FSX_MOUNT_PATH,
        "ckpt_mount": CKPT_MOUNT,
    }

    try:
        # Non-blocking: return immediately (timeout=0)
        # Docs: run_deployment blocks by default; set timeout=0 to return right away. 
        flow_run = run_deployment(
            name=PREFECT_DEPLOYMENT_PATH,
            parameters=params,
            timeout=0,
        )
        st.success(f"Submitted Prefect flow run: {getattr(flow_run, 'id', flow_run)}")
        st.toast("Flow submitted to Prefect (push-based).", icon="✅")
    except Exception as e:
        st.error(f"Submission failed: {e}")
