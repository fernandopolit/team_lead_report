"""
Data loaders for the coaching report.

  load_csv / load_latest  — read from a local CSV file
  load_from_sheets        — read from a Google Sheet via the Sheets API

Authentication for Google Sheets uses Application Default Credentials (ADC):
  - On GCP (Cloud Run): automatic via the attached service account
  - Locally: run `gcloud auth application-default login` first
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from data.schema import load_and_validate

_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def _resolve_dir(csv_dir: str, base_dir: str | None = None) -> Path:
    """
    Resolve csv_dir relative to base_dir (defaults to cwd).
    Also expands ~ and environment variables.
    """
    path = Path(os.path.expandvars(os.path.expanduser(csv_dir)))
    if not path.is_absolute():
        base = Path(base_dir) if base_dir else Path.cwd()
        path = (base / path).resolve()
    return path


def find_latest_csv(csv_dir: str, base_dir: str | None = None) -> Path:
    """Return the path of the most recently modified CSV in csv_dir."""
    directory = _resolve_dir(csv_dir, base_dir)
    if not directory.exists():
        raise FileNotFoundError(f"CSV directory not found: {directory}")

    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    return max(csv_files, key=lambda p: p.stat().st_mtime)


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load and validate a single CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    return load_and_validate(df)


def load_latest(csv_dir: str, base_dir: str | None = None) -> tuple[pd.DataFrame, Path]:
    """
    Find the latest CSV in csv_dir and return (DataFrame, path).
    base_dir is used to resolve relative csv_dir paths.
    """
    path = find_latest_csv(csv_dir, base_dir)
    df = load_csv(path)
    return df, path


def load_from_sheets(sheet_id: str, tab_name: str = "Sheet1") -> pd.DataFrame:
    """
    Read all rows from a Google Sheet tab and return a validated DataFrame.

    Authentication uses Application Default Credentials (ADC):
      - On GCP (Cloud Run): automatic via the attached service account
      - Locally: run `gcloud auth application-default login` first

    Args:
        sheet_id: The Google Sheet ID (from the URL: /spreadsheets/d/<ID>/)
        tab_name: Worksheet tab name (default: "Sheet1")

    Returns:
        Validated pandas DataFrame with the same columns the CSV loader expects.

    Raises:
        RuntimeError: If ADC credentials are not available.
        ImportError: If gspread / google-auth are not installed.
    """
    try:
        import google.auth
        import google.auth.transport.requests
        import gspread
    except ImportError:
        raise ImportError(
            "Google packages are not installed. Run:\n"
            "  pip install gspread google-auth google-auth-oauthlib"
        )

    try:
        creds, _ = google.auth.default(scopes=_SHEETS_SCOPES)
    except Exception as exc:
        raise RuntimeError(
            "Google credentials not found.\n"
            "To authenticate locally, run:\n"
            "  gcloud auth application-default login\n"
            f"Original error: {exc}"
        ) from exc

    # Refresh credentials so the token is valid before passing to gspread
    request = google.auth.transport.requests.Request()
    creds.refresh(request)

    gc = gspread.Client(auth=creds)
    gc.session = google.auth.transport.requests.AuthorizedSession(creds)

    try:
        spreadsheet = gc.open_by_key(sheet_id)
    except gspread.exceptions.APIError as exc:
        raise RuntimeError(
            f"Could not open Google Sheet '{sheet_id}'.\n"
            "Make sure your Google account has Viewer access to the sheet.\n"
            f"Original error: {exc}"
        ) from exc

    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        available = [ws.title for ws in spreadsheet.worksheets()]
        raise ValueError(
            f"Tab '{tab_name}' not found in sheet '{sheet_id}'.\n"
            f"Available tabs: {available}"
        )

    records = worksheet.get_all_records(numericise_ignore=["all"])
    if not records:
        raise ValueError(
            f"No data found in tab '{tab_name}' of sheet '{sheet_id}'."
        )

    df = pd.DataFrame(records)
    return load_and_validate(df)
