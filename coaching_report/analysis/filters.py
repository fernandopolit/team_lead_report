"""
Apply market and contact-reason filters to the raw dataset.
Both filters are case-insensitive and configurable.
"""
from __future__ import annotations

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    country_code: str = "ES",
    contact_reason_contains: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """
    Return rows matching:
      • country_code  (exact, case-insensitive)
      • contact_reason_l4  (substring, case-insensitive) — skipped if None
      • optional date range on contact_creation_timestamp

    Raises ValueError if no rows remain after filtering.
    """
    mask = df["country_code"].str.upper() == country_code.upper()

    if contact_reason_contains:
        mask &= df["contact_reason_l4"].str.lower().str.contains(
            contact_reason_contains.lower(), na=False
        )

    if date_start:
        mask &= df["contact_creation_timestamp"] >= pd.Timestamp(date_start, tz="UTC")
    if date_end:
        mask &= df["contact_creation_timestamp"] <= pd.Timestamp(date_end, tz="UTC")

    filtered = df[mask].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows matched filters: country_code='{country_code}', "
            f"contact_reason_contains='{contact_reason_contains}', "
            f"date_start={date_start}, date_end={date_end}"
        )

    return filtered
