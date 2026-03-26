"""
Google Drive uploader — uploads the generated HTML report to a shared folder.

Supports two filename modes (controlled by DRIVE_FILENAME_MODE env var):
  overwrite  (default) → CX_Coaching_Report_{market}.html
             BPO bookmarks one stable URL; file is updated in place each run.
  dated               → CX_Coaching_Report_{market}_YYYYMMDD.html
             Keeps full history; BPO receives a new link each day.

Authentication uses Application Default Credentials (ADC):
  - On GCP (Cloud Run): automatic via the attached service account
  - Locally: run `gcloud auth application-default login` first
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path


_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _get_credentials():
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=_DRIVE_SCOPES)
        return creds
    except ImportError:
        raise ImportError(
            "google-auth is not installed. "
            "Run: pip install google-auth google-api-python-client"
        )
    except Exception as exc:
        raise RuntimeError(
            "Google credentials not found. Run:\n"
            "  gcloud auth application-default login\n"
            f"Original error: {exc}"
        ) from exc


def drive_filename(market: str, run_date: date, mode: str = "overwrite") -> str:
    """Return the Drive filename based on the configured mode."""
    date_str = run_date.strftime("%Y%m%d")
    if mode == "dated":
        return f"MX_Coaching_Report_{market}_{date_str}.html"
    return f"MX_Coaching_Report_{market}.html"


def upload_to_drive(
    local_file_path: str | Path,
    folder_id: str,
    filename: str,
) -> tuple[str, str]:
    """
    Upload an HTML file to a Google Drive folder.

    If a file with the same name already exists in the folder it is updated
    in place (preserving the file ID and any existing sharing settings).
    If no file with that name exists a new one is created.

    Returns:
        (file_id, web_view_url)
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        raise ImportError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client"
        )

    creds = _get_credentials()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # Check whether the file already exists in the target folder
    safe_name = filename.replace("'", "\\'")
    query = (
        f"name='{safe_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    results = (
        service.files()
        .list(q=query, fields="files(id, name)", spaces="drive")
        .execute()
    )
    existing = results.get("files", [])

    media = MediaFileUpload(
        str(local_file_path),
        mimetype="text/html",
        resumable=False,
    )

    if existing:
        # Update the existing file in place — sharing settings are preserved
        file_id = existing[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
        ).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        result = (
            service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
        file_id = result["id"]

    web_view_url = f"https://drive.google.com/file/d/{file_id}/view"
    return file_id, web_view_url
