"""
Column definitions, type casting, and validation for the contacts dataset.
"""
from __future__ import annotations

import pandas as pd

# ── Required columns (must be present in every input file) ───────────────────
REQUIRED_COLUMNS: list[str] = [
    "stakeholder",
    "stakeholder_id",
    "country_code",
    "contact_id",
    "agent_email",
    "contact_creation_timestamp",
    "channel",
    "sat_score",
    "sat_free_text",
    "handling_time_mins",
    "contact_reason_l4",
    "is_actioned",
    "agent_actions",
    "order_id",
    "vertical_type",
    "vendor_name",
    "order_status",
    "is_cancelled_order",
    "cancellation_reason",
    "cancelled_at",
    "vehicle_type",
    "dispute_info",
    "content",
    "num_messages",
]


class SchemaError(Exception):
    """Raised when the dataset fails validation."""


def validate_columns(df: pd.DataFrame) -> None:
    """Raise SchemaError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to their expected types.
    Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()

    # Numeric
    df["sat_score"] = pd.to_numeric(df["sat_score"], errors="coerce")
    df["handling_time_mins"] = pd.to_numeric(df["handling_time_mins"], errors="coerce")
    df["num_messages"] = pd.to_numeric(df["num_messages"], errors="coerce")

    # Boolean columns — handle True/False strings as well as actual booleans
    for bool_col in ("is_actioned", "is_cancelled_order"):
        if df[bool_col].dtype == object:
            df[bool_col] = (
                df[bool_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
        df[bool_col] = df[bool_col].astype("boolean")

    # Timestamps
    df["contact_creation_timestamp"] = pd.to_datetime(
        df["contact_creation_timestamp"], errors="coerce", utc=True
    )
    df["cancelled_at"] = pd.to_datetime(
        df["cancelled_at"], errors="coerce", utc=True
    )

    # String columns — normalise NaN → empty string for easy filtering
    for str_col in (
        "country_code",
        "agent_email",
        "channel",
        "contact_reason_l4",
        "agent_actions",
        "vertical_type",
        "vendor_name",
        "order_status",
        "cancellation_reason",
        "vehicle_type",
        "content",
        "sat_free_text",
        "dispute_info",
    ):
        df[str_col] = df[str_col].fillna("").astype(str).str.strip()

    return df


def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate then cast. Returns clean DataFrame."""
    validate_columns(df)
    return cast_types(df)
