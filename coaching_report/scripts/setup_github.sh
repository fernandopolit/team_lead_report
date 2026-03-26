#!/usr/bin/env bash
# =============================================================================
# setup_github.sh — generate OAuth credentials for GitHub Actions
#
# No billing required. No service account required.
# Uses your personal Google OAuth credentials (refresh token) stored as a
# GitHub Actions secret — the same auth that works when you run locally.
#
# Run once from the coaching_report/ directory:
#   bash scripts/setup_github.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - A GitHub repository already created for this project
# =============================================================================
set -euo pipefail

ADC_FILE="$HOME/.config/gcloud/application_default_credentials.json"
SCOPES="https://www.googleapis.com/auth/spreadsheets.readonly,https://www.googleapis.com/auth/drive"

echo "================================================================="
echo " CX Coaching Report — GitHub Actions Setup"
echo " Using OAuth user credentials (no service account needed)"
echo "================================================================="
echo ""

# ── 1. Generate Application Default Credentials ───────────────────────────────
echo "▶ Opening browser for Google login…"
echo "  (this generates a refresh token saved to $ADC_FILE)"
echo ""
gcloud auth application-default login \
  --scopes="$SCOPES,https://www.googleapis.com/auth/cloud-platform"

if [[ ! -f "$ADC_FILE" ]]; then
  echo "Error: credentials file not found at $ADC_FILE"
  echo "Make sure 'gcloud auth application-default login' completed successfully."
  exit 1
fi

echo ""
echo "  ✓ Credentials saved to $ADC_FILE"

# ── 2. Summary and next steps ─────────────────────────────────────────────────
echo ""
echo "================================================================="
echo " Setup complete! Follow these steps to finish configuration:"
echo "================================================================="
echo ""
echo "1. Add the following secrets to your GitHub repository:"
echo "   (Settings → Secrets and variables → Actions → New repository secret)"
echo ""
echo "   GOOGLE_ADC_CREDENTIALS  ← paste the contents printed below"
echo "   GOOGLE_SHEET_ID         ← your Google Sheet ID"
echo "   DRIVE_FOLDER_ID         ← your Google Drive folder ID"
echo "   ANTHROPIC_API_KEY       ← your Anthropic API key (sk-ant-...)"
echo ""
echo "--- GOOGLE_ADC_CREDENTIALS (copy everything between the lines) ---"
cat "$ADC_FILE"
echo ""
echo "-----------------------------------------------------------------"
echo ""
echo "2. Add these optional repository variables:"
echo "   (Settings → Secrets and variables → Actions → Variables tab)"
echo ""
echo "   REPORT_MARKET          ES          (or MX, PT, etc.)"
echo "   GOOGLE_SHEET_TAB       Sheet1"
echo "   DRIVE_FILENAME_MODE    overwrite   (or 'dated' to keep history)"
echo "   REPORT_TIMEZONE        Europe/Madrid"
echo ""
echo "3. Share resources with YOUR Google account ($(gcloud config get account)):"
echo "   • Google Sheet  → already accessible (it's your account)"
echo "   • Drive folder  → already accessible (it's your account)"
echo ""
echo "4. Push the .github/workflows/daily_report.yml to your repo."
echo "   The report will run automatically at 06:00 UTC (07:00 Madrid) Mon–Fri."
echo ""
echo "   To trigger manually:"
echo "   GitHub → Actions → Daily CX Coaching Report → Run workflow"
echo ""
