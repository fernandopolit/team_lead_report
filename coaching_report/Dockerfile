FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# .dockerignore excludes credentials, .env files, outputs/, and __pycache__
COPY . .

# Cloud Run logs stdout/stderr — unbuffered output ensures logs appear in real time
ENV PYTHONUNBUFFERED=1

# All config comes from environment variables or Secret Manager at runtime.
# See deploy/cloudrun_job.yaml for the full env var specification.
ENTRYPOINT ["python", "main.py", "--source", "sheets"]
