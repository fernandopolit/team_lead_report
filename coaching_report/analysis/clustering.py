"""
Cluster assignment for PRIORITY_REVIEW agents.

Cluster A — Critical          actioned_rate_pct < 20
Cluster B — Half-Present      20 <= actioned_rate_pct < 50
Cluster C — Wrong Mix         50 <= actioned_rate_pct < 70  AND avg_csat still low
Cluster D — Active/Ineffective actioned_rate_pct >= 70
"""
from __future__ import annotations

import pandas as pd

CLUSTER_META = {
    "A": {
        "label": "Critical",
        "subtitle": "Near-Zero Action Rate",
        "color_class": "a",
        "description": (
            "These agents are almost never taking action. Customers contact support "
            "with a real problem and leave the chat having received nothing but "
            "apologies. The agent stays on the line but does not compensate, "
            "reassign, cancel, or call. The customer has no reason to give anything "
            "other than a 1."
        ),
    },
    "B": {
        "label": "Half-Present",
        "subtitle": "Low Action Rate + Limited Toolkit",
        "color_class": "b",
        "description": (
            "These agents are taking action on fewer than half their contacts. They "
            "respond to customers and sometimes act, but the majority of contacts end "
            "with no action. When they do act, they often use a single action without "
            "combining actions the way top performers do."
        ),
    },
    "C": {
        "label": "Wrong Mix",
        "subtitle": "Acting but Using Incomplete or Misaligned Actions",
        "color_class": "c",
        "description": (
            "These agents take action more than half the time — but the actions "
            "aren't landing. Some are over-relying on cancellations or chat-only "
            "compensation without combining actions. Some offer callbacks but don't "
            "follow through. The issue is action quality and combination, not "
            "just frequency."
        ),
    },
    "D": {
        "label": "Active/Ineffective",
        "subtitle": "High Action Rate, Quality Gap",
        "color_class": "d",
        "description": (
            "These agents take action frequently — the problem is execution quality. "
            "They are calling, compensating, and escalating. But the CSAT is still "
            "low. The failure is in how they communicate: scripted language, "
            "inability to close the loop, or compensation that isn't communicated "
            "clearly. These require call-review coaching rather than process coaching."
        ),
    },
}


def assign_cluster(actioned_rate_pct: float) -> str:
    """Return cluster letter A/B/C/D based on actioned rate."""
    if actioned_rate_pct < 20:
        return "A"
    if actioned_rate_pct < 50:
        return "B"
    if actioned_rate_pct < 70:
        return "C"
    return "D"


def add_clusters(agent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'cluster' column to PRIORITY_REVIEW agents.
    Non-PRIORITY_REVIEW agents get cluster = None.
    """
    df = agent_df.copy()
    df["cluster"] = None

    pr_mask = df["flag"] == "PRIORITY_REVIEW"
    df.loc[pr_mask, "cluster"] = df.loc[pr_mask, "actioned_rate_pct"].apply(
        assign_cluster
    )
    return df
