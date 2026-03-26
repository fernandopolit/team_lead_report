"""Tests for analysis/metrics.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_csv
from analysis.filters import apply_filters
from analysis.metrics import (
    compute_agent_metrics,
    compute_dataset_overview,
    flag_agents,
    parse_actions,
    count_tokens,
    compute_contact_response_times,
    classify_abandoned,
    DEFAULT_CLOSING_PHRASES,
)

SAMPLE = Path(__file__).parent / "sample_data.csv"


@pytest.fixture(scope="module")
def filtered_df():
    raw = load_csv(SAMPLE)
    return apply_filters(raw)


@pytest.fixture(scope="module")
def agent_metrics(filtered_df):
    return compute_agent_metrics(filtered_df)


@pytest.fixture(scope="module")
def overview(filtered_df):
    return compute_dataset_overview(filtered_df)


# ── parse_actions ─────────────────────────────────────────────────────────────

def test_parse_actions_normal():
    result = parse_actions("[has_compensation_done, has_customer_call]")
    assert result == ["has_compensation_done", "has_customer_call"]


def test_parse_actions_empty():
    assert parse_actions("[]") == []
    assert parse_actions("") == []
    assert parse_actions("nan") == []


def test_parse_actions_single():
    assert parse_actions("[has_cancel_done]") == ["has_cancel_done"]


# ── compute_dataset_overview ──────────────────────────────────────────────────

def test_overview_total_contacts(overview, filtered_df):
    assert overview["total_contacts"] == len(filtered_df)


def test_overview_csat_rate_between_0_and_1(overview):
    assert 0.0 <= overview["csat_response_rate"] <= 1.0


def test_overview_avg_csat_in_range(overview):
    assert 1.0 <= overview["overall_avg_csat"] <= 5.0


def test_overview_score_distribution_sums(overview):
    total = sum(overview["score_distribution"].values())
    assert total == overview["csat_responses"]


def test_overview_score1_rate_consistent(overview):
    expected = overview["score_distribution"][1] / overview["csat_responses"]
    assert abs(overview["score1_rate"] - expected) < 0.001


# ── compute_agent_metrics ─────────────────────────────────────────────────────

def test_agent_metrics_has_expected_columns(agent_metrics):
    for col in [
        "agent_email", "total_contacts", "avg_csat", "csat_responses",
        "low_csat_pct", "actioned_rate_pct", "comp_count", "call_count",
        "dominant_action_share",
    ]:
        assert col in agent_metrics.columns, f"Missing column: {col}"


def test_agent_metrics_all_agents_present(agent_metrics):
    handles = set(agent_metrics["agent_email"].str.split("@").str[0])
    expected = {
        "agent_a1", "agent_a2", "agent_b1", "agent_b2",
        "agent_c1", "agent_c2", "agent_d1", "agent_d2",
        "top_perf1", "top_perf2",
    }
    assert expected.issubset(handles), f"Missing agents: {expected - handles}"


def test_agent_actioned_rate_between_0_and_100(agent_metrics):
    assert (agent_metrics["actioned_rate_pct"] >= 0).all()
    assert (agent_metrics["actioned_rate_pct"] <= 100).all()


def test_low_csat_pct_between_0_and_100(agent_metrics):
    assert (agent_metrics["low_csat_pct"] >= 0).all()
    assert (agent_metrics["low_csat_pct"] <= 100).all()


def test_dominant_action_share_between_0_and_1(agent_metrics):
    assert (agent_metrics["dominant_action_share"] >= 0).all()
    assert (agent_metrics["dominant_action_share"] <= 1.0).all()


# ── csat_4_5_pct ─────────────────────────────────────────────────────────────

def test_agent_metrics_has_csat_4_5_pct(agent_metrics):
    assert "csat_4_5_pct" in agent_metrics.columns


def test_csat_4_5_pct_range(agent_metrics):
    valid = agent_metrics["csat_4_5_pct"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_csat_4_5_pct_all_high_scores():
    """If every scored contact is 4 or 5, csat_4_5_pct == 100."""
    import pandas as pd
    from analysis.metrics import compute_agent_metrics
    from data.schema import cast_types

    rows = []
    for i in range(5):
        rows.append({
            "agent_email": "x@test.com", "contact_id": str(i),
            "sat_score": 5.0 if i % 2 == 0 else 4.0,
            "is_actioned": True, "handling_time_mins": 5.0,
            "agent_actions": "[]", "content": "", "num_messages": 3,
            "country_code": "ES", "contact_reason_l4": "check order status",
            "order_status": "DELIVERED", "cancellation_reason": "",
            "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
            **{c: "" for c in [
                "stakeholder","stakeholder_id","channel","sat_free_text",
                "order_id","vertical_type","vendor_name","is_cancelled_order",
                "cancelled_at","vehicle_type","dispute_info",
            ]},
        })
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df)
    assert result.iloc[0]["csat_4_5_pct"] == 100.0


def test_csat_4_5_pct_no_high_scores():
    """If no scored contact is 4 or 5, csat_4_5_pct == 0."""
    import pandas as pd
    from analysis.metrics import compute_agent_metrics
    from data.schema import cast_types

    rows = []
    for i in range(5):
        rows.append({
            "agent_email": "y@test.com", "contact_id": str(i),
            "sat_score": 1.0,
            "is_actioned": False, "handling_time_mins": 5.0,
            "agent_actions": "[]", "content": "", "num_messages": 3,
            "country_code": "ES", "contact_reason_l4": "check order status",
            "order_status": "DELIVERED", "cancellation_reason": "",
            "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
            **{c: "" for c in [
                "stakeholder","stakeholder_id","channel","sat_free_text",
                "order_id","vertical_type","vendor_name","is_cancelled_order",
                "cancelled_at","vehicle_type","dispute_info",
            ]},
        })
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df)
    assert result.iloc[0]["csat_4_5_pct"] == 0.0


def test_csat_4_5_pct_zero_responses():
    """Agents with 0 CSAT responses get csat_4_5_pct = None."""
    import pandas as pd
    from analysis.metrics import compute_agent_metrics
    from data.schema import cast_types

    rows = [{
        "agent_email": "z@test.com", "contact_id": "1",
        "sat_score": None,
        "is_actioned": False, "handling_time_mins": 5.0,
        "agent_actions": "[]", "content": "", "num_messages": 3,
        "country_code": "ES", "contact_reason_l4": "check order status",
        "order_status": "DELIVERED", "cancellation_reason": "",
        "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
        **{c: "" for c in [
            "stakeholder","stakeholder_id","channel","sat_free_text",
            "order_id","vertical_type","vendor_name","is_cancelled_order",
            "cancelled_at","vehicle_type","dispute_info",
        ]},
    }]
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df)
    assert result.iloc[0]["csat_4_5_pct"] is None


def test_overview_has_overall_csat_4_5_pct(overview):
    assert "overall_csat_4_5_pct" in overview
    val = overview["overall_csat_4_5_pct"]
    assert val is None or (0.0 <= val <= 100.0)


def test_agent_a1_has_low_actioned_rate(agent_metrics):
    a1 = agent_metrics[agent_metrics["agent_email"] == "agent_a1@glovo.com"]
    assert len(a1) == 1
    assert a1.iloc[0]["actioned_rate_pct"] < 20


def test_top_perf1_has_high_csat(agent_metrics):
    tp = agent_metrics[agent_metrics["agent_email"] == "top_perf1@glovo.com"]
    assert len(tp) == 1
    assert tp.iloc[0]["avg_csat"] >= 3.0


# ── flag_agents ───────────────────────────────────────────────────────────────

def test_flag_agents_produces_expected_flags(agent_metrics, overview):
    flagged = flag_agents(
        agent_metrics,
        overall_avg_csat=overview["overall_avg_csat"],
        min_csat_responses=10,
    )
    flags = set(flagged["flag"].unique())
    assert flags.issubset({"PRIORITY_REVIEW", "TOP_PERFORMER", "NORMAL"})


def test_priority_review_agents_have_low_csat(agent_metrics, overview):
    flagged = flag_agents(
        agent_metrics,
        overall_avg_csat=overview["overall_avg_csat"],
        low_csat_offset=0.30,
        min_csat_responses=10,
    )
    pr = flagged[flagged["flag"] == "PRIORITY_REVIEW"]
    threshold = overview["overall_avg_csat"] - 0.30
    assert (pr["avg_csat"] <= threshold).all()


def test_top_performers_have_high_csat(agent_metrics, overview):
    flagged = flag_agents(
        agent_metrics,
        overall_avg_csat=overview["overall_avg_csat"],
        top_performer_offset=0.30,
        min_csat_responses=10,
    )
    tp = flagged[flagged["flag"] == "TOP_PERFORMER"]
    threshold = overview["overall_avg_csat"] + 0.30
    assert (tp["avg_csat"] >= threshold).all()


# ── compute_contact_response_times ───────────────────────────────────────────

def _ts(hms: str) -> str:
    """Helper: build a transcript line with a fixed date."""
    return f"2026-03-20 {hms}"


def _line(speaker: str, hms: str, msg: str = "hello") -> str:
    return f'> {speaker} ({_ts(hms)}): "{msg}"'


def test_response_time_single_turn():
    """Single Stakeholder→Agent turn returns that gap."""
    content = f"{_line('Stakeholder', '10:00:00')}\n{_line('Agent', '10:00:30')}"
    gaps = compute_contact_response_times(content)
    assert gaps == [30.0]


def test_response_time_gap_over_600_excluded():
    """Gaps > 600s are excluded."""
    content = f"{_line('Stakeholder', '10:00:00')}\n{_line('Agent', '10:11:00')}"
    gaps = compute_contact_response_times(content)
    assert gaps == []


def test_response_time_gap_under_3_excluded():
    """Gaps < 3s are excluded (likely automated messages)."""
    content = f"{_line('Stakeholder', '10:00:00')}\n{_line('Agent', '10:00:01')}"
    gaps = compute_contact_response_times(content)
    assert gaps == []


def test_response_time_mixed_valid_invalid():
    """Only valid gaps (3–600s) are included in the average."""
    lines = [
        _line('Stakeholder', '10:00:00'),
        _line('Agent', '10:00:30'),   # 30s — valid
        _line('Stakeholder', '10:01:00'),
        _line('Agent', '10:01:01'),   # 1s — too short, excluded
        _line('Stakeholder', '10:02:00'),
        _line('Agent', '10:12:30'),   # 630s — too long, excluded
        _line('Stakeholder', '10:13:00'),
        _line('Agent', '10:13:45'),   # 45s — valid
    ]
    gaps = compute_contact_response_times('\n'.join(lines))
    assert gaps == [30.0, 45.0]


def test_response_time_microseconds_handled():
    """Timestamps with microseconds parse correctly."""
    content = (
        '> Stakeholder (2026-03-20 10:00:00.000000): "hi"\n'
        '> Agent (2026-03-20 10:00:45.000000): "hello"'
    )
    gaps = compute_contact_response_times(content)
    assert gaps == [45.0]


def test_response_time_empty_content():
    """Empty or null content returns an empty list without raising."""
    assert compute_contact_response_times("") == []
    assert compute_contact_response_times("   ") == []


def test_response_time_all_case_channel_returns_none():
    """Agents with only Case-channel contacts get avg_response_time_secs = None."""
    from data.schema import cast_types
    rows = []
    for i in range(3):
        rows.append({
            "agent_email": "rt_agent@test.com", "contact_id": str(i),
            "sat_score": 3.0, "is_actioned": True, "handling_time_mins": 5.0,
            "agent_actions": "[]", "content": "", "num_messages": 3,
            "country_code": "ES", "contact_reason_l4": "check order status",
            "order_status": "DELIVERED", "cancellation_reason": "",
            "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
            "channel": "Case",  # ← not Chat
            **{c: "" for c in [
                "stakeholder", "stakeholder_id", "sat_free_text",
                "order_id", "vertical_type", "vendor_name", "is_cancelled_order",
                "cancelled_at", "vehicle_type", "dispute_info",
            ]},
        })
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df)
    assert result.iloc[0]["avg_response_time_secs"] is None


def test_response_time_multiple_contacts_averaged():
    """Per-contact averages are themselves averaged at the agent level."""
    from data.schema import cast_types

    # Contact 1: one 20s gap → per-contact avg = 20s
    # Contact 2: one 40s gap → per-contact avg = 40s
    # Agent avg = (20 + 40) / 2 = 30s
    c1 = '> Stakeholder (2026-03-20 10:00:00): "hi"\n> Agent (2026-03-20 10:00:20): "hello"'
    c2 = '> Stakeholder (2026-03-20 11:00:00): "hi"\n> Agent (2026-03-20 11:00:40): "hello"'

    rows = []
    for i, content in enumerate([c1, c2]):
        rows.append({
            "agent_email": "rt_multi@test.com", "contact_id": str(i),
            "sat_score": 3.0, "is_actioned": True, "handling_time_mins": 5.0,
            "agent_actions": "[]", "content": content, "num_messages": 3,
            "country_code": "ES", "contact_reason_l4": "check order status",
            "order_status": "DELIVERED", "cancellation_reason": "",
            "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
            "channel": "Chat",
            **{c: "" for c in [
                "stakeholder", "stakeholder_id", "sat_free_text",
                "order_id", "vertical_type", "vendor_name", "is_cancelled_order",
                "cancelled_at", "vehicle_type", "dispute_info",
            ]},
        })
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df)
    assert result.iloc[0]["avg_response_time_secs"] == 30.0


# ── classify_abandoned ────────────────────────────────────────────────────────

PHRASES = ["¿puedo ayudarte con algo más", "no dudes en contactarnos", "feel free to contact"]

def _conv(*turns) -> str:
    """Build a minimal transcript string from (speaker, HH:MM:SS, message) tuples."""
    return '\n'.join(
        f'> {spk} (2026-03-20 {t}): "{msg}"'
        for spk, t, msg in turns
    )


def test_classify_last_stakeholder_is_no_reply():
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:00:30', 'hello'),
        ('Stakeholder', '10:01:00', 'where is my order?'),
    )
    assert classify_abandoned(content, False, 1.0, PHRASES) == 'no_reply'


def test_classify_closing_phrase_not_abandoned():
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:00:30', '¿Puedo ayudarte con algo más? Gracias.'),
    )
    assert classify_abandoned(content, False, 1.0, PHRASES) is None


def test_classify_disengaged_no_action_score1():
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:00:30', 'I will check this'),
    )
    assert classify_abandoned(content, False, 1.0, PHRASES) == 'disengaged'


def test_classify_disengaged_false_positive_guard_actioned():
    """Guard: is_actioned=True → not disengaged even if no closing phrase and low CSAT."""
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:00:30', 'I will check this'),
    )
    assert classify_abandoned(content, True, 5.0, PHRASES) is None


def test_classify_late_response_customer_silent():
    """Agent responds after >300 s and customer doesn't reply again."""
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:05:01', 'sorry for the delay'),  # 301 s gap
    )
    # is_actioned=True prevents Type 2; sat_score=None triggers Type 3
    assert classify_abandoned(content, True, None, PHRASES) == 'late_response'


def test_classify_late_response_customer_reengaged_not_abandoned():
    """Customer re-engages after a late agent response → not abandoned."""
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:05:01', 'sorry'),       # 301 s gap
        ('Stakeholder', '10:05:30', 'ok'),           # re-engaged
        ('Agent',       '10:05:45', 'feel free to contact us'),  # closing phrase
    )
    assert classify_abandoned(content, False, 2.0, PHRASES) is None


def test_classify_no_reply_takes_precedence():
    """Last message is Stakeholder → no_reply, not disengaged (priority check)."""
    content = _conv(
        ('Agent',       '10:00:00', 'hello'),
        ('Stakeholder', '10:00:30', 'hi'),
        # last is Stakeholder; Type 2 would trigger if last were Agent
    )
    assert classify_abandoned(content, False, 1.0, PHRASES) == 'no_reply'


def test_classify_closing_phrase_case_insensitive():
    content = _conv(
        ('Stakeholder', '10:00:00', 'hi'),
        ('Agent',       '10:00:30', '¿PUEDO AYUDARTE CON ALGO MÁS?'),
    )
    assert classify_abandoned(content, False, 1.0, PHRASES) is None


def test_classify_null_content_returns_none():
    assert classify_abandoned(None, False, 1.0, PHRASES) is None
    assert classify_abandoned('', False, 1.0, PHRASES) is None


def test_agent_zero_abandoned_contacts_rate_is_zero():
    """All contacts have closing phrases → abandoned_rate_pct == 0.0 (not None)."""
    from data.schema import cast_types

    closing_phrase_content = '> Stakeholder (2026-03-20 10:00:00): "hi"\n> Agent (2026-03-20 10:00:30): "no dudes en contactarnos"'
    rows = []
    for i in range(3):
        rows.append({
            "agent_email": "clean@test.com", "contact_id": str(i),
            "sat_score": 1.0, "is_actioned": False, "handling_time_mins": 5.0,
            "agent_actions": "[]", "content": closing_phrase_content, "num_messages": 2,
            "country_code": "ES", "contact_reason_l4": "check order status",
            "order_status": "DELIVERED", "cancellation_reason": "",
            "contact_creation_timestamp": "2026-03-20T10:00:00+00:00",
            **{c: "" for c in [
                "stakeholder", "stakeholder_id", "channel", "sat_free_text",
                "order_id", "vertical_type", "vendor_name", "is_cancelled_order",
                "cancelled_at", "vehicle_type", "dispute_info",
            ]},
        })
    df = cast_types(pd.DataFrame(rows))
    result = compute_agent_metrics(df, closing_phrases=PHRASES)
    assert result.iloc[0]["abandoned_rate_pct"] == 0.0
