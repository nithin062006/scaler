# Hugging Face Space Dockerfile.
# Mirrors the root Dockerfile, exists separately because HF Spaces looks for
# the Dockerfile inside the Space root by default.

FROM python:3.11-slim

WORKDIR /app

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
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn env.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
