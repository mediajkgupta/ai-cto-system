# sandbox.Dockerfile
# Minimal Python image used as the execution sandbox.
# Built on demand; not strictly required since docker run uses python:3.11-slim directly.

FROM python:3.11-slim

# Create non-root user for extra safety
RUN useradd --create-home --shell /bin/bash sandbox
USER sandbox
WORKDIR /app

# No packages pre-installed — each generated project installs its own deps
CMD ["python", "main.py"]
