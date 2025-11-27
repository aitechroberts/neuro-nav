# app.py
import os
import json
from typing import List, Optional

# =============================================================================
# SECRETS BOOTSTRAP (must run BEFORE importing Prefect)
# Fetch PREFECT_API_KEY from AWS Secrets Manager and set as env var
# =============================================================================
def _bootstrap_secrets():
    """
    Fetch secrets from AWS Secrets Manager and set as environment variables.
    This must run before Prefect is imported, as Prefect reads PREFECT_API_KEY at import time.
    """
    import boto3
    from botocore.exceptions import ClientError
    
    region = os.getenv("AWS_REGION", "us-east-1")
    print(f"[bootstrap] Starting secrets bootstrap (region={region})")
    
    # Check if already set via ECS secrets injection
    existing_key = os.getenv("PREFECT_API_KEY")
    if existing_key:
        key_preview = existing_key[:8] + "..." if len(existing_key) > 8 else "***"
        print(f"[bootstrap] PREFECT_API_KEY already set (starts with: {key_preview})")
    else:
        secret_name = os.getenv("PREFECT_SECRET_NAME", "PrefectApiKey")
        print(f"[bootstrap] PREFECT_API_KEY not set, fetching from Secrets Manager ({secret_name})...")
        try:
            sm = boto3.client("secretsmanager", region_name=region)
            response = sm.get_secret_value(SecretId=secret_name)
            secret_value = response.get("SecretString", "")
            # Handle both plain string and JSON formats
            try:
                secret_json = json.loads(secret_value)
                # If JSON, look for common key names
                api_key = secret_json.get("PREFECT_API_KEY") or secret_json.get("api_key") or secret_json.get("key") or secret_value
            except json.JSONDecodeError:
                api_key = secret_value  # Plain string secret
            os.environ["PREFECT_API_KEY"] = api_key
            key_preview = api_key[:8] + "..." if len(api_key) > 8 else "***"
            print(f"[bootstrap] Loaded PREFECT_API_KEY from Secrets Manager (starts with: {key_preview})")
        except ClientError as e:
            print(f"[bootstrap] ERROR: Could not fetch Prefect secret '{secret_name}': {e}")
        except Exception as e:
            print(f"[bootstrap] ERROR: Unexpected error fetching Prefect secret: {e}")
    
    # Also log PREFECT_API_URL
    api_url = os.getenv("PREFECT_API_URL", "<not set>")
    print(f"[bootstrap] PREFECT_API_URL = {api_url}")

# Run bootstrap before any Prefect imports
_bootstrap_secrets()

import toml
import streamlit as st
import streamlit_shadcn_ui as ui
import streamlit.components.v1 as components
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

# Prefect (push-based trigger from the UI)
from prefect.deployments import run_deployment  # returns immediately with timeout=0  (docs)
# https://reference.prefect.io/prefect/deployments/  (run_deployment; timeout=0 to return immediately)

def load_themes():
    """Read the custom themes defined in `.streamlit/themes.toml`."""
    themes_file = os.path.join(".streamlit", "themes.toml")
    if os.path.exists(themes_file):
        with open(themes_file, "r", encoding="utf-8") as f:
            return toml.load(f)
    return {}


def update_theme(theme_dict):
    """Apply a theme by updating Streamlit's runtime config."""
    for key, value in theme_dict.items():
        st._config.set_option(f"theme.{key}", value)  # type: ignore[attr-defined]


THEMES = load_themes()
DEFAULT_THEME = "light" if "light" in THEMES else next(iter(THEMES), None)

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Data & GPU Jobs", page_icon="🧩", layout="wide")

# Ensure we start with a known theme
if DEFAULT_THEME and "current_theme_mode" not in st.session_state:
    update_theme(THEMES[DEFAULT_THEME])
    st.session_state.current_theme_mode = DEFAULT_THEME

# -----------------------------
# Theme toggle (user-facing)
# -----------------------------
col_toggle, _ = st.columns([1, 8])
with col_toggle:
    current_mode = st.session_state.get("current_theme_mode", DEFAULT_THEME or "light")
    switch_checked = current_mode == "dark"
    is_dark = ui.switch(
        default_checked=switch_checked,
        label="Dark Mode",
        key="theme_toggle",
    )

    selected_mode = "dark" if is_dark else "light"
    if (
        selected_mode != current_mode
        and selected_mode in THEMES
    ):
        update_theme(THEMES[selected_mode])
        st.session_state.current_theme_mode = selected_mode
        st.rerun()

dark_mode = st.session_state.get("current_theme_mode", "light") == "dark"

if dark_mode:
    # ---------------------------------------------------------
    # DARK MODE (NOOMO BEAT) CSS
    # ---------------------------------------------------------
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Oswald', sans-serif !important;
    }

    /* Custom Variables (accent + cards) */
    :root {
        --accent-color: #C084FC;
        --card-bg: #050505;
        --toggle-off: #ffffff;
        --toggle-on: #C084FC;
    }
    
    /* Sidebar Override */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color) !important;
        border-right: 1px solid #333 !important;
    }
    [data-testid="stSidebar"] * {
        color: #cccccc !important;
    }

    /* Title Styling */
    h1 {
        text-transform: uppercase;
        font-size: 4rem !important;
        letter-spacing: 2px;
        background: linear-gradient(180deg, #8A65AA 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 30px rgba(192, 132, 252, 0.4);
        font-weight: 700 !important;
        margin-bottom: 30px !important;
    }

    /* Card Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--accent-color) !important;
        background-color: var(--card-bg) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(192, 132, 252, 0.2), 0 0 0 1px rgba(192, 132, 252, 0.1) !important;
        padding: 1rem !important;
        margin-bottom: 2rem;
    }
    
    /* Card Headers */
    .card-header {
        background-color: var(--accent-color);
        color: white !important;
        font-size: 1.4rem;
        text-transform: uppercase;
        font-weight: 700;
        padding: 12px 20px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        border-bottom: 1px solid #a855f7;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        letter-spacing: 1px;
    }

    /* Input Fields */
    /* Specifically targeting the input wrapper to remove the default light gray background */
    .stTextInput > div > div {
        background-color: transparent !important;
    }
    
    .stTextInput > div > div > input {
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 20px !important;
    }
    
    /* File Uploader Button Override */
    [data-testid="stFileUploader"] button {
        background-color: transparent !important;
        color: var(--accent-color) !important;
        border: 1px solid var(--accent-color) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: var(--accent-color) !important;
        color: #000000 !important;
    }
    
    /* Toggle Switch Overrides */
    button[role="switch"][aria-checked="true"] {
        background-color: var(--toggle-on) !important;
        border-color: var(--toggle-on) !important;
    }
    button[role="switch"][aria-checked="false"] {
        background-color: var(--toggle-off) !important;
        border-color: var(--toggle-off) !important;
    }
    button[role="switch"] span {
        background-color: #000000 !important;
    }
    
    /* Direct Upload Card (HTML Component) Override */
    .direct-upload-card {
        border-color: var(--accent-color) !important;
        background-color: #0F172A !important; 
        color: #ffffff !important;
    }
    .direct-upload-header {
        color: var(--accent-color) !important;
    }
    input[type="submit"] {
        background-color: var(--accent-color) !important;
    }

    /* =======================
       Input Field Overrides
       ======================= */
    .stTextInput > div > div {
        background-color: transparent !important;
    }
    .stTextInput > div > div > input {
        background-color: #1a1524 !important;  /* slightly lighter than card */
        color: #ffffff !important;
        border: 1px solid var(--accent-color) !important;
        border-radius: 14px !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.35) inset;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    # ---------------------------------------------------------
    # LIGHT MODE (CMU BRAND) CSS
    # ---------------------------------------------------------
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Oswald', sans-serif !important;
    }

    /* Custom Variables */
    :root {
        --accent-color: #C41230;
        --card-bg: #FFFFFF;
        --toggle-off: #FFFFFF;
        --toggle-on: #000000;
    }
    
    /* Sidebar Override (Light) */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color) !important;
        border-right: 1px solid #ccc !important;
    }

    /* Title Styling - CMU Style */
    h1 {
        text-transform: uppercase;
        font-size: 4rem !important;
        color: var(--accent-color) !important; /* Carnegie Red Title */
        background: none !important;
        -webkit-text-fill-color: var(--accent-color) !important;
        text-shadow: none !important;
        font-weight: 700 !important;
        margin-bottom: 30px !important;
    }

    /* Card Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--accent-color) !important; /* Red Border */
        background-color: var(--card-bg) !important; /* White Bg */
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        padding: 1rem !important;
        margin-bottom: 2rem;
    }
    
    /* Card Headers */
    .card-header {
        background-color: var(--accent-color); /* Red Header */
        color: white !important;
        font-size: 1.4rem;
        text-transform: uppercase;
        font-weight: 700;
        padding: 12px 20px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        border-bottom: none;
        letter-spacing: 1px;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 20px !important;
    }
    
    /* File Uploader Button in Light Mode */
    [data-testid="stFileUploader"] button {
        background-color: transparent !important;
        color: var(--accent-color) !important;
        border: 1px solid var(--accent-color) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: var(--accent-color) !important;
        color: #ffffff !important;
    }
    
    /* Toggle Switch Overrides */
    button[role="switch"][aria-checked="true"] {
        background-color: var(--toggle-on) !important; /* Black */
        border-color: var(--toggle-on) !important;
    }
    button[role="switch"][aria-checked="false"] {
        background-color: var(--toggle-off) !important; /* White */
        border-color: #ccc !important; /* Grey border for visibility on white/grey bg */
    }
    button[role="switch"] span {
        background-color: #000000 !important; /* Black dot */
    }
    
    /* Direct Upload Card (HTML Component) Override */
    /* Since HTML component styles are inline, we'll handle them dynamically below */

    /* =======================
       Input Field Overrides
       ======================= */
    .stTextInput > div > div {
        background-color: transparent !important;
    }
    .stTextInput > div > div > input {
        background-color: #fbfbfb !important;  /* lighter than gray card */
        color: #000000 !important;
        border: 1px solid var(--accent-color) !important;
        border-radius: 14px !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) inset;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Data & GPU Jobs")

# ... (Rest of app logic same as previous)
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
DATASETS_BUCKET = os.getenv("DATASETS_BUCKET") or f"datasets-{ACCOUNT_ID}-{AWS_REGION}"
# Conceptually this bucket is used for evaluations; keep name for infra compatibility.
EVALUATIONS_BUCKET = os.getenv("EVALUATIONS_BUCKET", DATASETS_BUCKET)

ECR_REPOSITORY = os.getenv("ECR_REPOSITORY") or f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/gpu-jobs"
ECR_REPO_NAME = ECR_REPOSITORY.split("/")[-1]

# Public ECR (preferred): region is us-east-1 for ecr-public
ECR_PUBLIC_REPOSITORY_URI = os.getenv("ECR_PUBLIC_REPOSITORY_URI", "public.ecr.aws/r5i3x3r0/gpu-jobs")
ECR_PUBLIC_REPO_NAME = ECR_PUBLIC_REPOSITORY_URI.split("/")[-1]
ECR_PUBLIC_REGION = "us-east-1"

BATCH_JOB_QUEUE = os.getenv("BATCH_JOB_QUEUE", "gpu-batch-gpu-queue")
BATCH_JOB_DEFINITION_ARN = os.getenv("BATCH_JOB_DEFINITION_ARN", "")  # if you want to override


# IMPORTANT: These must match your Prefect v3 deployment "flow_name/deployment_name" paths
# Flow names use hyphens in Prefect (gpu-pipeline, evaluate-results)
# The UI dynamically selects which deployment to use based on user choices:
#   - upload-only: upload new data, no batch job
#   - run-existing: use existing data, run batch job
#   - upload-and-run: upload new data AND run batch job
#   - evaluate-existing: evaluation only
DEPLOYMENT_UPLOAD_ONLY = os.getenv("DEPLOYMENT_UPLOAD_ONLY", "gpu-pipeline/upload-only")
DEPLOYMENT_RUN_EXISTING = os.getenv("DEPLOYMENT_RUN_EXISTING", "gpu-pipeline/run-existing")
DEPLOYMENT_UPLOAD_AND_RUN = os.getenv("DEPLOYMENT_UPLOAD_AND_RUN", "gpu-pipeline/upload-and-run")
DEPLOYMENT_EVALUATE = os.getenv("DEPLOYMENT_EVALUATE", "evaluate-results/evaluate-existing")
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
def ecr_public_client():
    # ecr-public is a global service in us-east-1
    return boto3.client("ecr-public", region_name=ECR_PUBLIC_REGION, config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

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
def list_s3_prefixes(bucket: str, delimiter: str = "/") -> List[str]:
    """List top-level 'folders' (common prefixes) in an S3 bucket."""
    s3 = s3_client()
    prefixes: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Delimiter=delimiter):
        for cp in page.get("CommonPrefixes", []) or []:
            prefix = cp.get("Prefix", "")
            if prefix:
                prefixes.append(prefix.rstrip("/"))
    return prefixes

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

@st.cache_data(show_spinner=False, ttl=60)
def list_ecr_public_image_tags(repo_name: str) -> List[str]:
    ecrp = ecr_public_client()
    details = []
    for page in ecrp.get_paginator("describe_images").paginate(repositoryName=repo_name):
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
    
    # Prefect API status indicator
    prefect_key = os.getenv("PREFECT_API_KEY", "")
    prefect_url = os.getenv("PREFECT_API_URL", "")
    if prefect_key and prefect_url:
        st.success(f"✓ Prefect connected ({prefect_key[:8]}...)")
    elif prefect_key:
        st.warning("⚠ API key set but no URL")
    else:
        st.error("✗ PREFECT_API_KEY not set!")
    
    st.text_input("AWS Region", value=AWS_REGION, disabled=True)
    st.text_input("Raw Bucket", value=RAW_BUCKET or "<not set>", disabled=True)
    st.text_input("Checkpoints Bucket", value=CHECKPOINTS_BUCKET or "<not set>", disabled=True)
    st.text_input("Evaluations Bucket", value=EVALUATIONS_BUCKET or "<not set>", disabled=True)
    st.text_input("ECR Public Repo", value=ECR_PUBLIC_REPOSITORY_URI, disabled=True)
    st.text_input("Batch Queue", value=BATCH_JOB_QUEUE, disabled=True)
    st.text_input("Job Def ARN (override)", value=(BATCH_JOB_DEFINITION_ARN or "<not set>"), disabled=True)
    st.caption("Prefect Deployments (auto-selected):")
    st.caption(f"  • Upload+Run: {DEPLOYMENT_UPLOAD_AND_RUN}")
    st.caption(f"  • Run Existing: {DEPLOYMENT_RUN_EXISTING}")
    st.caption(f"  • Evaluate: {DEPLOYMENT_EVALUATE}")

# -----------------------------
# Helper for dynamic HTML styling (Direct Uploads)
# -----------------------------
def get_direct_upload_style(is_dark: bool):
    if is_dark:
        return """
        border: 1px solid #C084FC;
        background-color: #0F172A;
        color: #ffffff;
        """, """
        color: #C084FC;
        """, """
        font-size: 0.9rem;
        color: #cbd5e1;
        """, """
        background-color: #C084FC;
        color: white;
        """
    else:
        # Light Mode (CMU)
        return """
        border: 1px solid #C41230;
        background-color: #ffffff;
        color: #000000;
        """, """
        color: #C41230;
        """, """
        font-size: 0.9rem;
        color: #6D6E71;
        """, """
        background-color: #C41230;
        color: white;
        """

# -----------------------------
# Step 1: choose data
# -----------------------------
with st.container(border=True):
    # Card Header
    st.markdown('<div class="card-header">1) Choose data source</div>', unsafe_allow_html=True)

    data_mode_h = ui.tabs(options=["Upload new data", "Select existing data"], default_value="Upload new data", key="data_mode")

    upload_key: Optional[str] = None
    existing_key: Optional[str] = None
    uploaded_file = None  # Streamlit UploadedFile object
    uploaded_file_size: Optional[int] = None
    file_mime: Optional[str] = None

    if data_mode_h == "Upload new data":
        use_direct_upload = ui.switch(default_checked=False, label="Use browser-to-S3 (presigned) upload", key="use_direct_upload")
        st.markdown("**Destination key prefix in raw bucket**")
        upload_prefix = ui.input(default_value="", placeholder="replica/office0", key="upload_prefix")
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
            st.markdown("**Uploaded filename** (must match the file you choose below)")
            uploaded_filename_for_direct = ui.input(default_value="", placeholder="office0.zip", key="uploaded_filename_for_direct")
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
                
                # STYLED DIRECT UPLOAD FORM (Dynamic)
                card_style, header_style, info_style, btn_style = get_direct_upload_style(dark_mode)
                
                form_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');
    .direct-upload-card {{
        {card_style}
        padding: 20px;
        border-radius: 8px;
        font-family: 'Oswald', sans-serif;
        margin-top: 10px;
    }}
    .direct-upload-header {{
        {header_style}
        font-weight: 700;
        margin-bottom: 12px;
        font-size: 1.2rem;
        text-transform: uppercase;
    }}
    .direct-upload-info {{
        {info_style}
        margin-top: 8px;
    }}
    input[type="submit"] {{
        {btn_style}
        border: none;
        padding: 8px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 700;
        text-transform: uppercase;
        font-family: 'Oswald', sans-serif;
        transition: filter 0.2s;
    }}
    input[type="submit"]:hover {{
        filter: brightness(1.1);
    }}
    input[type="file"] {{
        color: inherit;
        font-family: 'Oswald', sans-serif;
    }}
    </style>
    <div class="direct-upload-card">
      <div class="direct-upload-header">Direct S3 upload</div>
      <form action="{form_action}" method="POST" enctype="multipart/form-data" target="_blank" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
        {form_fields_html}
        <label style="font-size:1rem;">Choose file:
          <input type="file" name="file" />
        </label>
        <input type="submit" value="Upload directly to S3"/>
      </form>
      <p class="direct-upload-info">After the upload completes (opens in a new tab), return and click Submit.</p>
      <p class="direct-upload-info">Object key will be <code>{(upload_prefix.rstrip('/') + '/' if upload_prefix else '')}$&#123;filename&#125;</code></p>
      <p class="direct-upload-info">Max 8GB via single POST.</p>
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
        st.markdown("**Existing key prefix to list**")
        prefix = ui.input(default_value="", key="list_prefix")
        try:
            keys = list_s3_objects(RAW_BUCKET, prefix)
        except ClientError as e:
            st.error(f"S3 list failed: {e}")
            keys = []
        st.markdown("**Choose an existing object**")
        existing_key = ui.select(options=keys, key="existing_key_select") if keys else None

# -----------------------------
# Step 2: Batch options
# -----------------------------
with st.container(border=True):
    st.markdown('<div class="card-header">2) Batch processing</div>', unsafe_allow_html=True)
    
    run_batch = ui.switch(default_checked=True, label="Run GPU batch job", key="run_batch")
    run_evaluation_after = ui.switch(default_checked=False, label="Run evaluation after batch job", key="run_evaluation_after")
    run_evaluation_only = ui.switch(default_checked=False, label="Run evaluation only (no batch job)", key="run_evaluation_only")
    selected_tag: Optional[str] = None
    output_folder: Optional[str] = None
    override_image = ui.switch(default_checked=False, label="Override Job Definition image with selected ECR tag", key="override_image")

    if run_batch or run_evaluation_only:
        # ----- IMAGE TAG SELECTOR -----
        st.markdown("**Choose GPU image tag (Public ECR)**")
        tags: List[str] = []
        try:
            tags = list_ecr_public_image_tags(ECR_PUBLIC_REPO_NAME)
        except ClientError as e:
            st.error(f"ECR public list failed: {e}")
        except Exception as e:
            st.warning(f"Could not list ECR public tags: {e}")
        
        if tags:
            selected_tag = ui.select(options=tags, key="selected_tag")
        else:
            st.warning("No image tags found in public ECR. Enter a tag manually:")
            selected_tag = ui.input(default_value="latest", key="selected_tag_manual")
        
        # ----- OUTPUT FOLDER SELECTOR -----
        st.markdown("**Output destination folder in finished bucket**")
        st.caption(f"Results (point cloud, semantic snapshot, etc.) will be uploaded to s3://{FINISHED_BUCKET}/<folder>/")
        
        # List existing folders (prefixes) in finished bucket
        try:
            finished_prefixes = list_s3_prefixes(FINISHED_BUCKET)
        except Exception as e:
            st.warning(f"Could not list finished bucket prefixes: {e}")
            finished_prefixes = []
        
        use_existing_folder = ui.switch(default_checked=False, label="Use existing folder", key="use_existing_output_folder")
        
        if use_existing_folder and finished_prefixes:
            output_folder = ui.select(options=finished_prefixes, key="output_folder_select")
        else:
            output_folder = ui.input(default_value="", placeholder="experiment_001", key="output_folder_input")
        
        if output_folder:
            st.info(f"Output will go to: s3://{FINISHED_BUCKET}/{output_folder.strip('/')}/")

    if run_evaluation_only and run_batch:
        st.warning("Both 'Run GPU batch job' and 'Run evaluation only' are enabled. "
                   "The Submit button will require you to choose one or the other.")

# -----------------------------
# Step 3: Optional checkpoint upload (no selection, not passed to Batch)
# -----------------------------
with st.container(border=True):
    st.markdown('<div class="card-header">3) Optional: Upload a checkpoint file</div>', unsafe_allow_html=True)
    
    do_ckpt_upload = ui.switch(default_checked=False, label="Upload a model checkpoint to checkpoints bucket", key="do_ckpt_upload")
    ckpt_upload_key: Optional[str] = None
    ckpt_uploaded_file = None
    ckpt_uploaded_file_size: Optional[int] = None
    ckpt_mime: Optional[str] = None

    if do_ckpt_upload:
        ckpt_use_direct_upload = ui.switch(default_checked=False, label="Use browser-to-S3 (presigned) upload for checkpoint", key="ckpt_use_direct_upload")
        st.markdown("**Destination key prefix in checkpoints bucket**")
        ckpt_prefix = ui.input(default_value="", placeholder="checkpoints/exp1", key="ckpt_prefix")
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
            st.markdown("**Uploaded checkpoint filename** (must match the file you choose below)")
            ckpt_uploaded_filename_for_direct = ui.input(
                default_value="", placeholder="checkpoint.pt", key="ckpt_uploaded_filename_for_direct"
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
                
                # STYLED CHECKPOINT UPLOAD FORM (Dynamic)
                card_style, header_style, info_style, btn_style = get_direct_upload_style(dark_mode)
                
                ckpt_form_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');
    .direct-upload-card {{
        {card_style}
        padding: 20px;
        border-radius: 8px;
        font-family: 'Oswald', sans-serif;
        margin-top: 10px;
    }}
    .direct-upload-header {{
        {header_style}
        font-weight: 700;
        margin-bottom: 12px;
        font-size: 1.2rem;
        text-transform: uppercase;
    }}
    .direct-upload-info {{
        {info_style}
        margin-top: 8px;
    }}
    input[type="submit"] {{
        {btn_style}
        border: none;
        padding: 8px 20px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 700;
        text-transform: uppercase;
        font-family: 'Oswald', sans-serif;
        transition: filter 0.2s;
    }}
    input[type="submit"]:hover {{
        filter: brightness(1.1);
    }}
    input[type="file"] {{
        color: inherit;
        font-family: 'Oswald', sans-serif;
    }}
    </style>
    <div class="direct-upload-card">
      <div class="direct-upload-header">Direct S3 upload (checkpoints)</div>
      <form action="{ckpt_form_action}" method="POST" enctype="multipart/form-data" target="_blank" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
        {ckpt_form_fields_html}
        <label style="font-size:1rem;">Choose file:
          <input type="file" name="file" />
        </label>
        <input type="submit" value="Upload checkpoint to S3"/>
      </form>
      <p class="direct-upload-info">After the upload completes (opens in a new tab), return and click Submit.</p>
      <p class="direct-upload-info">Object key will be <code>{(ckpt_prefix.rstrip('/') + '/' if ckpt_prefix else '')}$&#123;filename&#125;</code></p>
      <p class="direct-upload-info">Max 8GB via single POST.</p>
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

    # Evaluation-only path: skip batch pipeline and call evaluation deployment directly.
    if run_evaluation_only and not run_batch:
        # For evaluation-only runs, the input must already exist in the finished bucket as a folder.
        st.markdown("**Choose folder in finished bucket to evaluate**")
        try:
            eval_folders = list_s3_prefixes(FINISHED_BUCKET)
        except Exception as e:
            st.warning(f"Could not list finished bucket folders: {e}")
            eval_folders = []
        
        if eval_folders:
            eval_input_folder = ui.select(options=eval_folders, key="eval_input_folder_select")
        else:
            st.warning("No folders found. Enter folder name manually:")
            eval_input_folder = ui.input(default_value="", placeholder="experiment_001", key="eval_input_folder_manual")
        
        if not eval_input_folder:
            st.error("Please choose or enter a folder in the finished bucket to evaluate.")
            st.stop()

        eval_params = {
            "input_folder": eval_input_folder.strip("/") if eval_input_folder else "default",
            "finished_bucket": FINISHED_BUCKET,
            "evaluations_bucket": None,  # let Prefect Variable resolve
            "batch_job_queue": BATCH_JOB_QUEUE or None,
            "evaluation_job_definition_arn": BATCH_JOB_DEFINITION_ARN or None,
            "ecr_repo": ECR_PUBLIC_REPOSITORY_URI,
            "ecr_tag": (selected_tag or ""),
            "override_image": bool(override_image),
            "output_folder": f"eval-{eval_input_folder.strip('/')}" if eval_input_folder else "evaluation",
        }

        try:
            flow_run = run_deployment(
                name=DEPLOYMENT_EVALUATE,
                parameters=eval_params,
                timeout=0,
            )
            st.success(f"Submitted evaluation Prefect flow run: {getattr(flow_run, 'id', flow_run)}")
            st.toast("Evaluation flow submitted to Prefect (push-based).", icon="✅")
        except Exception as e:
            st.error(f"Evaluation submission failed: {e}")

        st.stop()

# -----------------------------
# Submit → Prefect deployment
# -----------------------------
if ui.button("Submit", key="submit_btn"):
    # Basic validation
    if run_evaluation_only and run_batch:
        st.error("Choose either 'Run GPU batch job' or 'Run evaluation only', not both.")
        st.stop()

    # Basic validation
    if not run_evaluation_only and data_mode_h == "Upload new data":
        if 'use_direct_upload' in locals() and use_direct_upload:
            if not uploaded_filename_for_direct:
                st.error("Please provide the uploaded filename to use.")
                st.stop()
        else:
            if uploaded_file is None:
                st.error("Please choose a file to upload.")
                st.stop()
    if (run_batch or run_evaluation_only) and (not output_folder):
        st.error("Please provide an output folder name.")
        st.stop()
    if (run_batch or run_evaluation_only) and override_image and not selected_tag:
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
        "run_evaluation_after": run_evaluation_after,
        "ecr_repo": ECR_PUBLIC_REPOSITORY_URI,
        "ecr_tag": (selected_tag or ""),
        "override_image": bool(override_image),
        "batch_job_queue": BATCH_JOB_QUEUE or None,                # let flow fallback to Variable if blank
        "batch_job_definition_arn": BATCH_JOB_DEFINITION_ARN or None,
        "output_folder": output_folder.strip("/") if output_folder else "default",
    }

    # Dynamically select deployment based on user choices
    is_upload = (data_mode_h == "Upload new data")
    if is_upload and run_batch:
        deployment_name = DEPLOYMENT_UPLOAD_AND_RUN
    elif is_upload and not run_batch:
        deployment_name = DEPLOYMENT_UPLOAD_ONLY
    elif not is_upload and run_batch:
        deployment_name = DEPLOYMENT_RUN_EXISTING
    else:
        # Fallback (existing data, no batch) - use run-existing but run_batch=False
        deployment_name = DEPLOYMENT_RUN_EXISTING

    try:
        # Debug: show what we're submitting
        st.info(f"Submitting to deployment: {deployment_name}")
        
        # Non-blocking: return immediately (timeout=0)
        # Docs: run_deployment blocks by default; set timeout=0 to return right away.
        flow_run = run_deployment(
            name=deployment_name,
            parameters=params,
            timeout=0,
        )
        st.success(f"Submitted Prefect flow run ({deployment_name}): {getattr(flow_run, 'id', flow_run)}")
        st.toast(f"Flow submitted: {deployment_name}", icon="✅")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Submission failed: {type(e).__name__}: {e}")
        st.code(error_details, language="python")
