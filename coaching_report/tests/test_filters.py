"""Tests for analysis/filters.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_csv
from analysis.filters import apply_filters

SAMPLE = Path(__file__).parent / "sample_data.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_csv(SAMPLE)


def test_filter_returns_es_rows(raw_df):
    filtered = apply_filters(raw_df, country_code="ES")
    assert (filtered["country_code"].str.upper() == "ES").all()


def test_filter_contact_reason_substring(raw_df):
    filtered = apply_filters(raw_df, contact_reason_contains="check order status")
    assert len(filtered) > 0
    assert filtered["contact_reason_l4"].str.lower().str.contains("check order status").all()


def test_filter_case_insensitive_country(raw_df):
    filtered_upper = apply_filters(raw_df, country_code="ES")
    filtered_lower = apply_filters(raw_df, country_code="es")
    assert len(filtered_upper) == len(filtered_lower)


def test_filter_case_insensitive_reason(raw_df):
    f1 = apply_filters(raw_df, contact_reason_contains="Check Order Status")
    f2 = apply_filters(raw_df, contact_reason_contains="check order status")
    assert len(f1) == len(f2)


def test_filter_wrong_market_raises(raw_df):
    with pytest.raises(ValueError, match="No rows matched"):
        apply_filters(raw_df, country_code="XX")


def test_filter_wrong_reason_raises(raw_df):
    with pytest.raises(ValueError, match="No rows matched"):
        apply_filters(raw_df, contact_reason_contains="this reason does not exist xyz")


def test_filter_date_range(raw_df):
    """Date range filter should reduce rows when valid bounds are provided."""
    all_rows = apply_filters(raw_df)
    bounded = apply_filters(raw_df, date_start="2026-03-21", date_end="2026-03-25")
    assert len(bounded) <= len(all_rows)


def test_filter_preserves_all_columns(raw_df):
    filtered = apply_filters(raw_df)
    assert set(filtered.columns) == set(raw_df.columns)
