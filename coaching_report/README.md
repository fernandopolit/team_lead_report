# CX Coaching Report

Automated weekly Team Leader coaching report — reads contact data, computes agent metrics, and generates a self-contained HTML report with LLM-assisted coaching copy.

## Architecture

```
Google Sheets → Cloud Run Job → Google Drive shared folder
                     ↑
          Cloud Scheduler (07:00 Mon–Fri, Madrid)
```

---

## Local Development

### Run against a local CSV (no GCP needed)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with the latest CSV in the default directory
python main.py

# Run with a specific file
python main.py --file path/to/data.csv

# Filter by market and contact reason
python main.py --market MX --contact-reason "accidental order"

# Rule-based coaching only (no LLM calls)
python main.py --no-llm
```

### Run against the live Google Sheet

```bash
# Authenticate with GCP (one-time, saves credentials locally)
gcloud auth application-default login

# Run against Google Sheets
python main.py --source sheets

# With explicit sheet ID (overrides GOOGLE_SHEET_ID env var)
GOOGLE_SHEET_ID=your_sheet_id python main.py --source sheets
```

### Build and test the Docker image locally

```bash
# Build
docker build -t cx-coaching-report .

# Run against a local CSV (mount the CSV directory)
docker run \
  -v "$(pwd)/../:/data" \
  -e REPORT_MARKET=ES \
  cx-coaching-report --source csv

# Run against Google Sheets (mount local GCP credentials)
docker run \
  -e GOOGLE_SHEET_ID=your_sheet_id \
  -e DRIVE_FOLDER_ID=your_folder_id \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  cx-coaching-report
```

---

## GCP Deployment

### First-time setup

Run the setup script once to provision all GCP resources:

```bash
export GCP_PROJECT_ID=your-project-id
bash scripts/setup_gcp.sh
```

This script:
1. Enables all required APIs
2. Creates the `coaching-report-sa` service account
3. Grants IAM roles
4. Creates Secret Manager secrets (with placeholder values)
5. Builds and pushes the Docker image via Cloud Build
6. Deploys the Cloud Run Job
7. Creates the Cloud Scheduler trigger (07:00 Mon–Fri)

After running, fill in the real secret values:

```bash
# Google Sheet ID (from sheet URL: /spreadsheets/d/<ID>/)
echo -n 'YOUR_SHEET_ID' \
  | gcloud secrets versions add cx-report-sheet-id --data-file=-

# Google Drive folder ID (from folder URL: /folders/<ID>)
echo -n 'YOUR_FOLDER_ID' \
  | gcloud secrets versions add cx-report-drive-folder-id --data-file=-

# Anthropic API key
echo -n 'sk-ant-...' \
  | gcloud secrets versions add cx-report-anthropic-api-key --data-file=-
```

Then share the resources with the service account:
- **Google Sheet** → Share with `coaching-report-sa@PROJECT_ID.iam.gserviceaccount.com` as **Viewer**
- **Google Drive folder** → Share with the same service account as **Editor**

### Manual trigger

```bash
gcloud run jobs execute cx-coaching-report --region=europe-west1
```

### Check logs

```bash
gcloud logging read \
  'resource.type=cloud_run_job AND resource.labels.job_name=cx-coaching-report' \
  --limit=50 --format="table(timestamp,textPayload)"
```

### Update the schedule

Edit `deploy/scheduler.yaml`, then:

```bash
gcloud scheduler jobs update http cx-coaching-report-daily \
  --schedule="0 8 * * 1-5" \
  --time-zone="Europe/Madrid" \
  --location=europe-west1
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_SHEET_ID` | Yes (cloud) | — | Sheet ID from the URL |
| `GOOGLE_SHEET_TAB` | No | `Sheet1` | Worksheet tab name |
| `DRIVE_FOLDER_ID` | Yes (cloud) | — | Drive folder ID from the URL |
| `DRIVE_FILENAME_MODE` | No | `overwrite` | `overwrite` (stable URL) or `dated` (keeps history) |
| `REPORT_MARKET` | No | `ES` | Market filter (country code) |
| `REPORT_CONTACT_REASON` | No | _(all)_ | Contact reason substring filter |
| `REPORT_TIMEZONE` | No | `Europe/Madrid` | Timezone for the run date in the report |
| `GCP_PROJECT_ID` | No | — | GCP project ID (logging only) |
| `ANTHROPIC_API_KEY` | Yes (LLM) | — | Anthropic API key for coaching copy |

Sensitive variables (`GOOGLE_SHEET_ID`, `DRIVE_FOLDER_ID`, `ANTHROPIC_API_KEY`) are stored in Secret Manager and injected at runtime — never hardcoded in the image or config files.

---

## Configuration (`config.yaml`)

Non-sensitive defaults. All values can be overridden by environment variables or CLI flags.

```yaml
filters:
  country_code: "ES"              # overridden by REPORT_MARKET
  contact_reason_contains: null   # null = all reasons; overridden by REPORT_CONTACT_REASON

coaching:
  use_llm: true
  llm_model: "claude-sonnet-4-6"
  llm_max_tokens: 1200
```

---

## Delivery modes

| Mode | Filename | Use case |
|---|---|---|
| `overwrite` (default) | `CX_Coaching_Report_ES.html` | BPO bookmarks one stable Drive URL, updated each morning |
| `dated` | `CX_Coaching_Report_ES_20260326.html` | Keeps history; team receives a fresh link each day |
