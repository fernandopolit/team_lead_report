#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh — one-time GCP provisioning for the CX Coaching Report
#
# Run once from the coaching_report/ directory:
#   bash scripts/setup_gcp.sh
#
# The script is idempotent — running it twice will not duplicate resources.
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - GCP_PROJECT_ID environment variable set, or passed as first argument
#   - Billing enabled on the project
# =============================================================================
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-${1:-}}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: set GCP_PROJECT_ID env var or pass the project ID as an argument."
  echo "  Usage: GCP_PROJECT_ID=my-project bash scripts/setup_gcp.sh"
  exit 1
fi

REGION="europe-west1"
SA_NAME="coaching-report-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
JOB_NAME="cx-coaching-report"
SCHEDULER_JOB="cx-coaching-report-daily"
IMAGE="gcr.io/${PROJECT_ID}/cx-coaching-report"

echo "================================================================="
echo " CX Coaching Report — GCP Setup"
echo " Project: $PROJECT_ID | Region: $REGION"
echo "================================================================="

gcloud config set project "$PROJECT_ID" --quiet

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo ""
echo "▶ Enabling APIs…"
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  sheets.googleapis.com \
  drive.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  --project="$PROJECT_ID"
echo "  ✓ APIs enabled"

# ── 2. Create service account (idempotent) ────────────────────────────────────
echo ""
echo "▶ Creating service account ${SA_EMAIL}…"
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
  echo "  ✓ Service account already exists — skipping"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="CX Coaching Report Runner" \
    --project="$PROJECT_ID"
  echo "  ✓ Service account created"
fi

# ── 3. Grant IAM roles ────────────────────────────────────────────────────────
echo ""
echo "▶ Granting IAM roles to ${SA_EMAIL}…"
for ROLE in \
  roles/secretmanager.secretAccessor \
  roles/run.invoker \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --condition=None \
    --quiet
  echo "  ✓ $ROLE"
done

# ── 4. Create secrets in Secret Manager (idempotent) ─────────────────────────
echo ""
echo "▶ Creating secrets in Secret Manager…"

create_secret_if_missing() {
  local SECRET_NAME="$1"
  local PLACEHOLDER="$2"
  if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "  ✓ Secret '$SECRET_NAME' already exists — skipping"
  else
    echo -n "$PLACEHOLDER" | gcloud secrets create "$SECRET_NAME" \
      --data-file=- \
      --project="$PROJECT_ID"
    echo "  ✓ Secret '$SECRET_NAME' created with placeholder value"
  fi
}

create_secret_if_missing "cx-report-sheet-id"          "REPLACE_WITH_SHEET_ID"
create_secret_if_missing "cx-report-drive-folder-id"   "REPLACE_WITH_FOLDER_ID"
create_secret_if_missing "cx-report-anthropic-api-key" "REPLACE_WITH_ANTHROPIC_API_KEY"

# Grant the service account access to each secret
for SECRET in cx-report-sheet-id cx-report-drive-folder-id cx-report-anthropic-api-key; do
  gcloud secrets add-iam-policy-binding "$SECRET" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="$PROJECT_ID" \
    --quiet
done
echo "  ✓ Secret access granted to service account"

# ── 5. Build and push Docker image ────────────────────────────────────────────
echo ""
echo "▶ Building and pushing Docker image via Cloud Build…"
gcloud builds submit \
  --config cloudbuild.yaml \
  --project="$PROJECT_ID" \
  .
echo "  ✓ Image pushed: ${IMAGE}:latest"

# ── 6. Deploy Cloud Run Job ───────────────────────────────────────────────────
echo ""
echo "▶ Deploying Cloud Run Job '${JOB_NAME}'…"
# Substitute project ID into the YAML and pipe to gcloud
sed "s/\${GCP_PROJECT_ID}/${PROJECT_ID}/g" deploy/cloudrun_job.yaml \
  | gcloud run jobs replace - \
      --region="$REGION" \
      --project="$PROJECT_ID"
echo "  ✓ Cloud Run Job deployed"

# ── 7. Create Cloud Scheduler job (idempotent) ────────────────────────────────
echo ""
echo "▶ Configuring Cloud Scheduler job '${SCHEDULER_JOB}'…"

# Get the Cloud Run Job execution URL
RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "$SCHEDULER_JOB" \
    --location="$REGION" \
    --project="$PROJECT_ID" &>/dev/null; then
  echo "  ✓ Scheduler job already exists — updating schedule only"
  gcloud scheduler jobs update http "$SCHEDULER_JOB" \
    --schedule="0 7 * * 1-5" \
    --time-zone="Europe/Madrid" \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --quiet
else
  gcloud scheduler jobs create http "$SCHEDULER_JOB" \
    --schedule="0 7 * * 1-5" \
    --time-zone="Europe/Madrid" \
    --uri="$RUN_URL" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL" \
    --max-retry-attempts=2 \
    --min-backoff=5m \
    --location="$REGION" \
    --project="$PROJECT_ID"
fi
echo "  ✓ Scheduler job configured: 07:00 Mon–Fri Europe/Madrid"

# ── 8. Summary and next steps ─────────────────────────────────────────────────
echo ""
echo "================================================================="
echo " Setup complete!"
echo "================================================================="
echo ""
echo "REQUIRED: Fill in the placeholder secrets before the first run:"
echo ""
echo "  # Your Google Sheet ID (from the sheet URL)"
echo "  echo -n 'YOUR_SHEET_ID' | gcloud secrets versions add cx-report-sheet-id --data-file=-"
echo ""
echo "  # Your Google Drive folder ID (from the folder URL)"
echo "  echo -n 'YOUR_FOLDER_ID' | gcloud secrets versions add cx-report-drive-folder-id --data-file=-"
echo ""
echo "  # Your Anthropic API key"
echo "  echo -n 'sk-ant-...' | gcloud secrets versions add cx-report-anthropic-api-key --data-file=-"
echo ""
echo "REQUIRED: Share resources with the service account (${SA_EMAIL}):"
echo "  • Google Sheet → Share as Viewer"
echo "  • Google Drive folder → Share as Editor"
echo ""
echo "To run the job manually:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION}"
echo ""
echo "To run locally against the live Google Sheet:"
echo "  gcloud auth application-default login"
echo "  GOOGLE_SHEET_ID=your_id DRIVE_FOLDER_ID=your_id python main.py --source sheets"
echo ""
echo "To check logs:"
echo "  gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}' --limit=50"
echo ""
