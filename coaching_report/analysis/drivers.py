"""
Dataset-wide low-CSAT driver analysis.

Computes avg CSAT / low_csat_pct broken down by:
  order_status, cancellation_reason, vehicle_type, vertical_type,
  handling_time bucket, num_messages bucket, channel.

Also flags underperforming vendors.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _bucket_series(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, right=False)


HT_EDGES = [0, 5, 10, 15, 20, 30, 60, float("inf")]
HT_LABELS = ["0–5", "5–10", "10–15", "15–20", "20–30", "30–60", "60+"]

MSG_EDGES = [0, 6, 11, 16, 21, 31, 51, float("inf")]
MSG_LABELS = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–50", "50+"]


def _group_stats(df: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    """Compute avg_csat and low_csat_pct grouped by `col`."""
    scored = df[df["sat_score"].notna()].copy()
    rows = []
    for val, grp in scored.groupby(col, observed=True):
        if len(grp) == 0:
            continue
        low = grp[grp["sat_score"] <= 2]
        rows.append(
            {
                col: str(val) if val else "(unknown)",
                "contacts": len(grp),
                "avg_csat": round(float(grp["sat_score"].mean()), 2),
                "low_csat_pct": round(len(low) / len(grp) * 100, 1),
            }
        )
    return sorted(rows, key=lambda r: r["contacts"], reverse=True)


def compute_drivers(df: pd.DataFrame) -> dict[str, Any]:
    """Return driver analysis dict."""
    work = df.copy()

    # Add bucket columns
    work["ht_bucket"] = _bucket_series(
        work["handling_time_mins"].fillna(0), HT_EDGES, HT_LABELS
    )
    work["msg_bucket"] = _bucket_series(
        work["num_messages"].fillna(0), MSG_EDGES, MSG_LABELS
    )

    drivers: dict[str, Any] = {
        "order_status": _group_stats(work, "order_status"),
        "cancellation_reason": _group_stats(
            work[work["cancellation_reason"] != ""], "cancellation_reason"
        ),
        "vehicle_type": _group_stats(
            work[work["vehicle_type"] != ""], "vehicle_type"
        ),
        "vertical_type": _group_stats(
            work[work["vertical_type"] != ""], "vertical_type"
        ),
        "channel": _group_stats(work, "channel"),
        "handling_time_bucket": _group_stats(work, "ht_bucket"),
        "num_messages_bucket": _group_stats(work, "msg_bucket"),
    }

    # Underperforming vendors
    scored = work[work["sat_score"].notna()]
    vendor_rows = []
    for vendor, grp in scored.groupby("vendor_name"):
        if not vendor or vendor == "":
            continue
        low = grp[grp["sat_score"] <= 2]
        vendor_rows.append(
            {
                "vendor_name": vendor,
                "contacts": len(grp),
                "avg_csat": round(float(grp["sat_score"].mean()), 2),
                "low_csat_pct": round(len(low) / len(grp) * 100, 1),
            }
        )
    drivers["underperforming_vendors"] = [
        v for v in vendor_rows
        if v["contacts"] >= 20 and v["avg_csat"] <= 2.20
    ]
    drivers["underperforming_vendors"].sort(key=lambda v: v["avg_csat"])

    return drivers
