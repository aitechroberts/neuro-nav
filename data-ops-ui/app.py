# app.py
import os
import json
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

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
uploaded_file = None  # Streamlit UploadedFile object
uploaded_file_size: Optional[int] = None
file_mime: Optional[str] = None

if data_mode_h == "Upload new data":
    use_direct_upload = st.toggle("Use browser-to-S3 (presigned) upload", value=False)
    upload_prefix = st.text_input("Destination key prefix in raw bucket", placeholder="replica/office0")
    if not use_direct_upload:
        up = st.file_uploader("Upload input file (e.g., Replica ZIP)")
        if up is not None:
            upload_key = f"{upload_prefix.rstrip('/')}/{up.name}"
            uploaded_file = up
            uploaded_file_size = getattr(up, "size", None)
            file_mime = getattr(up, "type", None)
            st.info(f"Will upload to s3://{RAW_BUCKET}/{upload_key}")
    else:
        st.caption("Direct upload uses a pre-signed POST so the browser uploads straight to S3 without passing through this server.")
        uploaded_filename_for_direct = st.text_input("Uploaded filename (must match the file you choose below)", placeholder="office0.zip")
        # Generate a presigned POST form that allows the browser to POST directly to S3
        try:
            key_prefix = f"{upload_prefix.rstrip('/')}/" if upload_prefix else ""
            key_template = f"{key_prefix}${{filename}}"
            post = s3_client().generate_presigned_post(
                Bucket=RAW_BUCKET,
                Key=key_template,
                Fields={
                    "key": key_template,
                },
                Conditions=[
                    ["starts-with", "$key", key_prefix],
                    ["content-length-range", 1, 8 * 1024 * 1024 * 1024],  # up to 8GB
                ],
                ExpiresIn=3600,
            )
            form_fields_html = "\n".join(
                [f'<input type="hidden" name="{k}" value="{v}"/>' for k, v in post["fields"].items()]
            )
            form_action = post["url"]
            form_html = f"""
<div style="padding:12px;border:1px solid #e6e9ef;border-radius:8px;background:#fafbfc;">
  <p style="margin:0 0 8px 0;font-weight:600;">Direct S3 upload</p>
  <form action="{form_action}" method="POST" enctype="multipart/form-data" target="_blank" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    {form_fields_html}
    <label style="font-size:0.9rem;">Choose file:
      <input type="file" name="file" style="margin-left:8px;" />
    </label>
    <input type="submit" value="Upload directly to S3" style="padding:6px 10px;"/>
  </form>
  <p style="margin:8px 0 0 0;font-size:0.9rem;">After the upload completes (opens in a new tab), return and click Submit.</p>
  <p style="margin:4px 0 0 0;font-size:0.9rem;color:#555;">Object key will be <code>{(upload_prefix.rstrip('/') + '/' if upload_prefix else '')}$&#123;filename&#125;</code></p>
  <p style="margin:4px 0 0 0;font-size:0.85rem;color:#777;">Max 8GB via single POST.</p>
  </div>
"""
            components.html(form_html, height=300)
            if uploaded_filename_for_direct:
                st.info(
                    f"After direct upload, this app will use s3://{RAW_BUCKET}/"
                    f"{(upload_prefix.rstrip('/') + '/' if upload_prefix else '')}{uploaded_filename_for_direct}"
                )
        except Exception as e:
            st.error(f"Failed to prepare direct upload form: {e}")
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
# Step 3: Optional checkpoint upload (no selection, not passed to Batch)
# -----------------------------
st.subheader("3) Optional: Upload a checkpoint file")
do_ckpt_upload = st.toggle("Upload a model checkpoint to checkpoints bucket", value=False)
ckpt_upload_key: Optional[str] = None
ckpt_uploaded_file = None
ckpt_uploaded_file_size: Optional[int] = None
ckpt_mime: Optional[str] = None

if do_ckpt_upload:
    ckpt_use_direct_upload = st.toggle("Use browser-to-S3 (presigned) upload for checkpoint", value=False)
    ckpt_prefix = st.text_input("Destination key prefix in checkpoints bucket", placeholder="checkpoints/exp1")
    if not ckpt_use_direct_upload:
        ckpt_up = st.file_uploader("Checkpoint file", key="ckpt_uploader")
        if ckpt_up is not None:
            ckpt_upload_key = f"{ckpt_prefix.rstrip('/')}/{ckpt_up.name}"
            ckpt_uploaded_file = ckpt_up
            ckpt_uploaded_file_size = getattr(ckpt_up, "size", None)
            ckpt_mime = getattr(ckpt_up, "type", None)
            st.info(f"Will upload to s3://{CHECKPOINTS_BUCKET}/{ckpt_upload_key}")
    else:
        st.caption("Direct upload uses a pre-signed POST so the browser uploads straight to S3 without passing through this server.")
        ckpt_uploaded_filename_for_direct = st.text_input(
            "Uploaded checkpoint filename (must match the file you choose below)", placeholder="checkpoint.pt"
        )
        try:
            ckpt_key_prefix = f"{ckpt_prefix.rstrip('/')}/" if ckpt_prefix else ""
            ckpt_key_template = f"{ckpt_key_prefix}${{filename}}"
            ckpt_post = s3_client().generate_presigned_post(
                Bucket=CHECKPOINTS_BUCKET,
                Key=ckpt_key_template,
                Fields={"key": ckpt_key_template},
                Conditions=[
                    ["starts-with", "$key", ckpt_key_prefix],
                    ["content-length-range", 1, 8 * 1024 * 1024 * 1024],
                ],
                ExpiresIn=3600,
            )
            ckpt_form_fields_html = "\n".join(
                [f'<input type="hidden" name="{k}" value="{v}"/>' for k, v in ckpt_post["fields"].items()]
            )
            ckpt_form_action = ckpt_post["url"]
            ckpt_form_html = f"""
<div style="padding:12px;border:1px solid #e6e9ef;border-radius:8px;background:#fafbfc;">
  <p style="margin:0 0 8px 0;font-weight:600;">Direct S3 upload (checkpoints)</p>
  <form action="{ckpt_form_action}" method="POST" enctype="multipart/form-data" target="_blank" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    {ckpt_form_fields_html}
    <label style="font-size:0.9rem;">Choose file:
      <input type="file" name="file" style="margin-left:8px;" />
    </label>
    <input type="submit" value="Upload checkpoint to S3" style="padding:6px 10px;"/>
  </form>
  <p style="margin:8px 0 0 0;font-size:0.9rem;">After the upload completes (opens in a new tab), return and click Submit.</p>
  <p style="margin:4px 0 0 0;font-size:0.9rem;color:#555;">Object key will be <code>{(ckpt_prefix.rstrip('/') + '/' if ckpt_prefix else '')}$&#123;filename&#125;</code></p>
  <p style="margin:4px 0 0 0;font-size:0.85rem;color:#777;">Max 8GB via single POST.</p>
  </div>
"""
            components.html(ckpt_form_html, height=300)
            if ckpt_uploaded_filename_for_direct:
                st.info(
                    f"After direct upload, this app will store your checkpoint at s3://{CHECKPOINTS_BUCKET}/"
                    f"{(ckpt_prefix.rstrip('/') + '/' if ckpt_prefix else '')}{ckpt_uploaded_filename_for_direct}"
                )
        except Exception as e:
            st.error(f"Failed to prepare direct checkpoint upload form: {e}")

st.markdown("---")

# -----------------------------
# Submit → Prefect deployment
# -----------------------------
if st.button("Submit"):
    # Basic validation
    if data_mode_h == "Upload new data":
        if 'use_direct_upload' in locals() and use_direct_upload:
            if not uploaded_filename_for_direct:
                st.error("Please provide the uploaded filename to use.")
                st.stop()
        else:
            if uploaded_file is None:
                st.error("Please choose a file to upload.")
                st.stop()
    if run_batch and (not output_name):
        st.error("Please provide an output artifact name.")
        st.stop()
    if run_batch and override_image and not selected_tag:
        st.error("You chose to override the image; please select an ECR tag.")
        st.stop()

    # If uploading a new file, stream it directly to S3 (multipart) from the UI server
    final_data_mode = "existing"
    final_existing_key = existing_key
    if data_mode_h == "Upload new data":
        if 'use_direct_upload' in locals() and use_direct_upload:
            # Direct upload path: assume user has uploaded via the presigned form; use the constructed key
            if upload_prefix:
                final_existing_key = f"{upload_prefix.rstrip('/')}/{uploaded_filename_for_direct}"
            else:
                final_existing_key = f"{uploaded_filename_for_direct}"
            st.success(f"Using uploaded object s3://{RAW_BUCKET}/{final_existing_key}")
        else:
            if not upload_key:
                st.error("Please provide a destination key prefix.")
                st.stop()
            try:
                st.info("Uploading to S3...")
                progress_bar = st.progress(0)
                status = st.empty()
                progress_state = {"bytes": 0}

                def _progress_hook(bytes_amount: int):
                    progress_state["bytes"] += bytes_amount
                    if uploaded_file_size and uploaded_file_size > 0:
                        pct = int(min(progress_state["bytes"] * 100 / uploaded_file_size, 100))
                        progress_bar.progress(pct)
                        status.text(f"Uploaded {progress_state['bytes'] / (1024*1024):.1f} MB / {uploaded_file_size / (1024*1024):.1f} MB")

                cfg = TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,
                    multipart_chunksize=8 * 1024 * 1024,   # 8MB parts to lower memory
                    max_concurrency=1,                     # single-threaded to avoid parallel buffers
                    use_threads=False,
                )
                # Ensure file pointer at start
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                extra_args = {"ContentType": file_mime} if file_mime else None
                if extra_args:
                    s3_client().upload_fileobj(
                        uploaded_file, RAW_BUCKET, upload_key, ExtraArgs=extra_args, Callback=_progress_hook, Config=cfg
                    )
                else:
                    s3_client().upload_fileobj(
                        uploaded_file, RAW_BUCKET, upload_key, Callback=_progress_hook, Config=cfg
                    )
                progress_bar.progress(100)
                status.text("Upload complete.")
                st.success(f"Uploaded s3://{RAW_BUCKET}/{upload_key}")
                final_existing_key = upload_key
            except Exception as e:
                st.error(f"S3 upload failed: {e}")
                st.stop()

    # Handle optional checkpoint upload (not passed to Batch)
    if do_ckpt_upload:
        if 'ckpt_use_direct_upload' in locals() and ckpt_use_direct_upload:
            if not ckpt_uploaded_filename_for_direct:
                st.error("Please provide the uploaded checkpoint filename to use.")
                st.stop()
            # For direct upload we assume user already uploaded via the form; no server-side action needed.
            ckpt_final_key = (
                f"{ckpt_prefix.rstrip('/')}/{ckpt_uploaded_filename_for_direct}" if ckpt_prefix else ckpt_uploaded_filename_for_direct
            )
            st.success(f"Using uploaded checkpoint s3://{CHECKPOINTS_BUCKET}/{ckpt_final_key}")
        else:
            if not ckpt_upload_key:
                st.error("Please provide a destination key prefix for the checkpoint.")
                st.stop()
            try:
                st.info("Uploading checkpoint to S3...")
                ckpt_progress = st.progress(0)
                ckpt_status = st.empty()
                ckpt_state = {"bytes": 0}

                def _ckpt_progress_hook(bytes_amount: int):
                    ckpt_state["bytes"] += bytes_amount
                    if ckpt_uploaded_file_size and ckpt_uploaded_file_size > 0:
                        pct = int(min(ckpt_state["bytes"] * 100 / ckpt_uploaded_file_size, 100))
                        ckpt_progress.progress(pct)
                        ckpt_status.text(
                            f"Uploaded {ckpt_state['bytes'] / (1024*1024):.1f} MB / {ckpt_uploaded_file_size / (1024*1024):.1f} MB"
                        )

                cfg_ckpt = TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,
                    multipart_chunksize=8 * 1024 * 1024,
                    max_concurrency=1,
                    use_threads=False,
                )
                try:
                    ckpt_uploaded_file.seek(0)  # type: ignore[union-attr]
                except Exception:
                    pass
                ckpt_extra_args = {"ContentType": ckpt_mime} if ckpt_mime else None
                if ckpt_extra_args:
                    s3_client().upload_fileobj(
                        ckpt_uploaded_file,  # type: ignore[arg-type]
                        CHECKPOINTS_BUCKET,
                        ckpt_upload_key,
                        ExtraArgs=ckpt_extra_args,
                        Callback=_ckpt_progress_hook,
                        Config=cfg_ckpt,
                    )
                else:
                    s3_client().upload_fileobj(
                        ckpt_uploaded_file,  # type: ignore[arg-type]
                        CHECKPOINTS_BUCKET,
                        ckpt_upload_key,
                        Callback=_ckpt_progress_hook,
                        Config=cfg_ckpt,
                    )
                ckpt_progress.progress(100)
                ckpt_status.text("Checkpoint upload complete.")
                st.success(f"Uploaded s3://{CHECKPOINTS_BUCKET}/{ckpt_upload_key}")
            except Exception as e:
                st.error(f"S3 checkpoint upload failed: {e}")
                st.stop()

    # Map UI → flow parameters (must match gpu_pipeline signature)
    params = {
        "data_mode": final_data_mode,
        "upload_key": upload_key,
        "upload_content_b64": None,  # large uploads are handled directly to S3 by the UI
        "content_type": file_mime if data_mode_h == "Upload new data" else None,
        "existing_key": final_existing_key,
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
