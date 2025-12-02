# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Application code and configuration (including custom themes)
COPY app.py flows/flows.py ./
COPY .streamlit ./.streamlit

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=8192 \
    STREAMLIT_SERVER_MAX_MESSAGE_SIZE=8192 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
