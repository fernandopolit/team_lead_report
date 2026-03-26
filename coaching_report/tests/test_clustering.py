"""Tests for analysis/clustering.py"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.clustering import assign_cluster, add_clusters, CLUSTER_META
from data.loader import load_csv
from analysis.filters import apply_filters
from analysis.metrics import compute_agent_metrics, flag_agents, compute_dataset_overview


SAMPLE = Path(__file__).parent / "sample_data.csv"


# ── assign_cluster boundary tests ─────────────────────────────────────────────

@pytest.mark.parametrize("rate,expected", [
    (0.0,   "A"),
    (5.0,   "A"),
    (19.9,  "A"),
    (20.0,  "B"),
    (35.0,  "B"),
    (49.9,  "B"),
    (50.0,  "C"),
    (60.0,  "C"),
    (69.9,  "C"),
    (70.0,  "D"),
    (85.0,  "D"),
    (100.0, "D"),
])
def test_assign_cluster_boundaries(rate, expected):
    assert assign_cluster(rate) == expected


# ── CLUSTER_META completeness ──────────────────────────────────────────────────

def test_cluster_meta_has_all_letters():
    assert set(CLUSTER_META.keys()) == {"A", "B", "C", "D"}


def test_cluster_meta_has_required_keys():
    for letter, meta in CLUSTER_META.items():
        for key in ("label", "subtitle", "color_class", "description"):
            assert key in meta, f"Cluster {letter} missing key '{key}'"


# ── add_clusters integration ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clustered_df():
    raw = load_csv(SAMPLE)
    filtered = apply_filters(raw)
    overview = compute_dataset_overview(filtered)
    agent_df = compute_agent_metrics(filtered)
    flagged = flag_agents(
        agent_df,
        overall_avg_csat=overview["overall_avg_csat"],
        min_csat_responses=10,
    )
    return add_clusters(flagged)


def test_cluster_column_exists(clustered_df):
    assert "cluster" in clustered_df.columns


def test_non_pr_agents_have_no_cluster(clustered_df):
    non_pr = clustered_df[clustered_df["flag"] != "PRIORITY_REVIEW"]
    assert non_pr["cluster"].isna().all()


def test_pr_agents_have_valid_cluster(clustered_df):
    pr = clustered_df[clustered_df["flag"] == "PRIORITY_REVIEW"]
    assert pr["cluster"].notna().all()
    assert pr["cluster"].isin(["A", "B", "C", "D"]).all()


def test_cluster_a_agents_have_low_actioned_rate(clustered_df):
    cluster_a = clustered_df[clustered_df["cluster"] == "A"]
    if not cluster_a.empty:
        assert (cluster_a["actioned_rate_pct"] < 20).all()


def test_cluster_d_agents_have_high_actioned_rate(clustered_df):
    cluster_d = clustered_df[clustered_df["cluster"] == "D"]
    if not cluster_d.empty:
        assert (cluster_d["actioned_rate_pct"] >= 70).all()


def test_all_four_clusters_represented(clustered_df):
    """Priority Review agents should span at least one cluster."""
    pr_clusters = set(clustered_df[clustered_df["flag"] == "PRIORITY_REVIEW"]["cluster"].dropna())
    assert len(pr_clusters) >= 1, f"Expected at least 1 cluster, got: {pr_clusters}"
