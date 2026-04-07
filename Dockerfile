# Accept base image as build argument — openenv push overrides this with the
# public HF base image automatically (ghcr.io/meta-pytorch/openenv-base:latest)
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

COPY envs/airen_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY src/openenv/core/ /app/src/openenv/core/
COPY envs/airen_env/ /app/envs/airen_env/

WORKDIR /app
ENV PYTHONPATH=/app/src:/app/envs
ENV MAX_CONCURRENT_ENVS=64

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "envs.airen_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
