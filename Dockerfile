# Container for the GraphForge OpenEnv server.
# Used by Hugging Face Spaces (Docker SDK) and by anyone running the env
# in production. Fast start, slim image, minimal runtime deps.

FROM python:3.11-slim

WORKDIR /app

# Install only what the server needs at runtime — training deps are not
# required to host the env.
COPY pyproject.toml ./
COPY graphforge ./graphforge
COPY env ./env
COPY openenv.yaml ./

RUN pip install --no-cache-dir \
    "pydantic>=2.6" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "httpx>=0.27" \
    "openenv-core>=0.1.0" \
    "pyyaml>=6.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# Hugging Face Spaces routes inbound traffic to whatever port the container
# binds; we read $PORT but default to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn env.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
