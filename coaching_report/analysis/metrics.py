"""
All agent-level and dataset-level metric computations.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

# ── Action token helpers ──────────────────────────────────────────────────────

# Known action tokens and friendly labels
ACTION_LABELS: dict[str, str] = {
    "has_compensation_done": "compensation",
    "has_customer_call": "customer call",
    "has_outbound_call": "outbound call",
    "has_vendor_call": "vendor call",
    "has_courier_call": "courier call",
    "has_cancel_done": "cancel",
    "confirm_cancel": "confirm cancel",
    "has_reassign_done": "reassign",
    "has_partial_refund_done": "partial refund",
    "has_address_change": "address change",
}

COMPENSATION_TOKENS = {"has_compensation_done"}
CALL_TOKENS = {"has_customer_call", "has_outbound_call"}
VENDOR_CALL_TOKENS = {"has_vendor_call", "has_courier_call"}
CANCEL_TOKENS = {"has_cancel_done", "confirm_cancel"}
REASSIGN_TOKENS = {"has_reassign_done"}

# Parses "> Speaker (YYYY-MM-DD HH:MM:SS" lines in chat transcripts (compiled once)
TS_PATTERN = re.compile(
    r'^> (Agent|Stakeholder) \((\d{4}-\d{2}-\d{2} '
    r'\d{2}:\d{2}:\d{2})'
)

# Matches "Soy FirstName LastName" in agent transcript lines (Spanish intro pattern)
_SOY_RE = re.compile(
    r'\bSoy\s+([A-ZÁÉÍÓÚ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚ][a-záéíóúüñ]+)+)',
    re.UNICODE,
)

DEFAULT_CLOSING_PHRASES: list[str] = [
    "¿puedo ayudarte con algo más",
    "no dudes en contactarnos",
    "que tengas un buen",
    "gracias por contactar",
    "en breve recibirás una encuesta",
    "hemos intentado ponernos en contacto",
    "voy a dar por terminada esta conversación",
    "no logramos localizarte",
    "is there anything else",
    "feel free to contact",
]


def _last_agent_block_text(content: str) -> str:
    """Return all text from the final '> Agent (...)' line to end-of-content, lowercased."""
    lines = content.split('\n')
    last_agent_idx = -1
    for i, line in enumerate(lines):
        m = TS_PATTERN.match(line.strip())
        if m and m.group(1) == 'Agent':
            last_agent_idx = i
    if last_agent_idx < 0:
        return ''
    return '\n'.join(lines[last_agent_idx:]).lower()


def classify_abandoned(
    content: str | None,
    is_actioned: bool,
    sat_score: float | None,
    closing_phrases: list[str],
) -> str | None:
    """
    Classify a contact's abandonment type. Priority: no_reply > disengaged > late_response.

    Type 1 (no_reply):   last message belongs to Stakeholder.
    Type 2 (disengaged): last is Agent, no closing phrase, has Stakeholder turn,
                         is_actioned=False, sat_score null or ≤ 2.
    Type 3 (late_response): Stakeholder→Agent gap > 300 s, customer does not
                             re-engage after, sat_score null or ≤ 2.

    Returns 'no_reply', 'disengaged', 'late_response', or None.
    """
    if not content or not isinstance(content, str) or not content.strip():
        return None

    msg_seq: list[tuple[str, datetime]] = []
    for line in content.split('\n'):
        m = TS_PATTERN.match(line.strip())
        if m:
            try:
                ts = datetime.strptime(m.group(2), '%Y-%m-%d %H:%M:%S')
                msg_seq.append((m.group(1), ts))
            except ValueError:
                pass

    if not msg_seq:
        return None

    # Type 1: last message is from Stakeholder
    if msg_seq[-1][0] == 'Stakeholder':
        return 'no_reply'

    has_stakeholder = any(s == 'Stakeholder' for s, _ in msg_seq)
    low_or_null_sat = sat_score is None or sat_score <= 2.0

    # Type 2: last is Agent, no closing phrase, customer present, not actioned, low/null sat
    if has_stakeholder and not is_actioned and low_or_null_sat:
        last_agent_text = _last_agent_block_text(content)
        if not any(phrase in last_agent_text for phrase in closing_phrases):
            return 'disengaged'

    # Type 3: >300 s Stakeholder→Agent gap after which customer doesn't re-engage, low/null sat
    if low_or_null_sat and has_stakeholder:
        last_sh_ts: datetime | None = None
        for i, (speaker, ts) in enumerate(msg_seq):
            if speaker == 'Stakeholder':
                last_sh_ts = ts
            elif speaker == 'Agent' and last_sh_ts is not None:
                gap = (ts - last_sh_ts).total_seconds()
                last_sh_ts = None
                if gap > 300:
                    has_followup = any(
                        msg_seq[j][0] == 'Stakeholder' for j in range(i + 1, len(msg_seq))
                    )
                    if not has_followup:
                        return 'late_response'

    return None


def compute_contact_response_times(content: str) -> list[float]:
    """
    Parse a single chat transcript and return valid Stakeholder→Agent response
    gaps in seconds.  Valid range: 3–600 s inclusive.

    Expects lines in the format:
      > Agent (YYYY-MM-DD HH:MM:SS[.ffffff]): "message"
      > Stakeholder (YYYY-MM-DD HH:MM:SS[.ffffff]): "message"

    The regex captures only up to whole seconds; microseconds are ignored.
    """
    if not content or not isinstance(content, str):
        return []

    messages: list[tuple[str, datetime]] = []
    for line in content.split('\n'):
        m = TS_PATTERN.match(line.strip())
        if m:
            try:
                ts = datetime.strptime(m.group(2), '%Y-%m-%d %H:%M:%S')
                messages.append((m.group(1), ts))
            except ValueError:
                pass

    gaps: list[float] = []
    last_stakeholder_ts: datetime | None = None
    for speaker, ts in messages:
        if speaker == 'Stakeholder':
            last_stakeholder_ts = ts
        elif speaker == 'Agent' and last_stakeholder_ts is not None:
            gap = (ts - last_stakeholder_ts).total_seconds()
            if 3.0 <= gap <= 600.0:
                gaps.append(gap)
            last_stakeholder_ts = None  # consumed — wait for next Stakeholder turn
    return gaps


def parse_actions(actions_str: str) -> list[str]:
    """
    Parse a stringified action list like '[has_compensation_done, has_customer_call]'
    into a Python list of token strings.
    """
    if not actions_str or actions_str in ("[]", "nan", ""):
        return []
    cleaned = actions_str.strip("[]").strip()
    if not cleaned:
        return []
    return [t.strip() for t in cleaned.split(",") if t.strip()]


def count_tokens(series: pd.Series) -> Counter:
    """Count all action tokens across a Series of action strings."""
    counter: Counter = Counter()
    for s in series:
        counter.update(parse_actions(str(s)))
    return counter


def _token_count(token_set: set[str], counter: Counter) -> int:
    return sum(counter.get(t, 0) for t in token_set)


# ── Per-agent metrics ─────────────────────────────────────────────────────────

def compute_agent_metrics(df: pd.DataFrame, closing_phrases: list[str] | None = None) -> pd.DataFrame:
    """
    Compute agent-level metrics. Returns one row per agent_email.
    """
    rows: list[dict[str, Any]] = []
    if closing_phrases is None:
        closing_phrases = DEFAULT_CLOSING_PHRASES

    for agent_email, grp in df.groupby("agent_email", sort=False):
        scored = grp[grp["sat_score"].notna()]
        low_csat = scored[scored["sat_score"] <= 2]

        total = len(grp)
        csat_responses = len(scored)
        avg_csat = float(scored["sat_score"].mean()) if csat_responses > 0 else float("nan")
        low_csat_pct = (len(low_csat) / csat_responses * 100) if csat_responses > 0 else 0.0
        high_csat = scored[scored["sat_score"] >= 4]
        csat_4_5_pct = (
            round(len(high_csat) / csat_responses * 100, 1) if csat_responses > 0 else None
        )
        actioned_rate_pct = (
            grp["is_actioned"].fillna(False).astype(bool).sum() / total * 100
        )
        avg_ht = float(grp["handling_time_mins"].mean()) if total > 0 else 0.0

        # Score distribution — snap floats to nearest integer bucket 1-5
        score_dist: dict[int, int] = {k: 0 for k in range(1, 6)}
        for score, cnt in scored["sat_score"].value_counts().items():
            try:
                bucket = int(round(float(score)))
                if bucket in score_dist:
                    score_dist[bucket] += int(cnt)
            except (ValueError, TypeError):
                pass

        # Action counts
        action_counter = count_tokens(grp["agent_actions"])
        top_actions = dict(action_counter.most_common(5))
        total_action_tokens = sum(action_counter.values())

        comp_count = _token_count(COMPENSATION_TOKENS, action_counter)
        call_count = _token_count(CALL_TOKENS, action_counter)
        # Count contacts where at least one call action appeared (avoids >100% rates)
        call_contact_count = int(
            grp["agent_actions"].apply(
                lambda s: bool(set(parse_actions(str(s))) & CALL_TOKENS)
            ).sum()
        )
        vendor_call_count = _token_count(VENDOR_CALL_TOKENS, action_counter)
        cancel_count = _token_count(CANCEL_TOKENS, action_counter)
        reassign_count = _token_count(REASSIGN_TOKENS, action_counter)

        dominant_action = ""
        dominant_action_share = 0.0
        if total_action_tokens > 0 and action_counter:
            dominant_action, dominant_count = action_counter.most_common(1)[0]
            dominant_action_share = dominant_count / total_action_tokens

        # Low-CSAT context
        low_csat_order_statuses: dict[str, int] = (
            low_csat["order_status"].value_counts().head(5).to_dict()
            if not low_csat.empty
            else {}
        )
        low_csat_cancel_reasons: dict[str, int] = {}
        if not low_csat.empty:
            cancelled_lc = low_csat[low_csat["cancellation_reason"] != ""]
            low_csat_cancel_reasons = (
                cancelled_lc["cancellation_reason"].value_counts().head(3).to_dict()
            )

        delivered_mask = low_csat["order_status"].str.upper() == "DELIVERED"
        low_csat_delivered_pct = (
            delivered_mask.sum() / len(low_csat) if not low_csat.empty else 0.0
        )

        late_delivery_count = int(
            low_csat[
                low_csat["cancellation_reason"].str.upper() == "LATE_DELIVERY"
            ].shape[0]
        )

        # Score-1 and score-2 conversation examples (for tabbed layout)
        score1_with_content = grp[
            (grp["sat_score"] == 1) & (grp["content"].str.len() > 100)
        ]
        score2_with_content = grp[
            (grp["sat_score"] == 2) & (grp["content"].str.len() > 100)
        ]

        # Avg response time — Chat contacts only
        chat_contacts = grp[grp["channel"].str.strip().str.lower() == "chat"] \
            if "channel" in grp.columns else pd.DataFrame()
        per_contact_avgs: list[float] = []
        for _, cr in chat_contacts.iterrows():
            cv = cr.get("content", "")
            if pd.isna(cv) or not str(cv).strip():
                continue
            contact_gaps = compute_contact_response_times(str(cv))
            if contact_gaps:
                per_contact_avgs.append(sum(contact_gaps) / len(contact_gaps))
        avg_response_time_secs = (
            round(sum(per_contact_avgs) / len(per_contact_avgs), 1)
            if per_contact_avgs else None
        )

        # Abandonment classification
        abandoned_types: list[str | None] = []
        for _, row in grp.iterrows():
            cv = row.get("content")
            ia_raw = row.get("is_actioned", False)
            ia = bool(ia_raw) if not pd.isna(ia_raw) else False
            ss = row.get("sat_score")
            sat_clean = None if pd.isna(ss) else float(ss)
            abandoned_types.append(
                classify_abandoned(
                    str(cv) if pd.notna(cv) and str(cv).strip() else None,
                    ia, sat_clean, closing_phrases,
                )
            )

        abandoned_count = sum(1 for t in abandoned_types if t is not None)
        no_reply_count = sum(1 for t in abandoned_types if t == 'no_reply')
        disengaged_count = sum(1 for t in abandoned_types if t == 'disengaged')
        late_resp_count = sum(1 for t in abandoned_types if t == 'late_response')
        abandoned_rate_pct = round(abandoned_count / total * 100, 1) if total else 0.0
        abandoned_no_reply_pct = round(no_reply_count / total * 100, 1) if total else 0.0
        abandoned_disengaged_pct = round(disengaged_count / total * 100, 1) if total else 0.0
        abandoned_late_response_pct = round(late_resp_count / total * 100, 1) if total else 0.0

        # Extract agent's real name from "Soy [First Last]" in transcript content
        agent_name: str | None = None
        for content_val in grp["content"].dropna():
            for line in str(content_val).split('\n'):
                m = _SOY_RE.search(line)
                if m:
                    candidate = m.group(1)
                    if len(candidate.split()) >= 2:  # require at least first + last name
                        agent_name = candidate
                        break
            if agent_name:
                break

        rows.append(
            {
                "agent_email": agent_email,
                "total_contacts": total,
                "avg_csat": round(avg_csat, 2),
                "csat_4_5_pct": csat_4_5_pct,
                "csat_responses": csat_responses,
                "low_csat_pct": round(low_csat_pct, 1),
                "avg_handling_time_mins": round(avg_ht, 1),
                "actioned_rate_pct": round(actioned_rate_pct, 1),
                "score_distribution": score_dist,
                "top_actions": top_actions,
                "action_counter": action_counter,
                "total_action_tokens": total_action_tokens,
                "comp_count": comp_count,
                "call_count": call_count,
                "call_contact_count": call_contact_count,
                "vendor_call_count": vendor_call_count,
                "cancel_count": cancel_count,
                "reassign_count": reassign_count,
                "dominant_action": dominant_action,
                "dominant_action_share": round(dominant_action_share, 3),
                "low_csat_order_statuses": low_csat_order_statuses,
                "low_csat_cancel_reasons": low_csat_cancel_reasons,
                "low_csat_delivered_pct": round(low_csat_delivered_pct, 3),
                "late_delivery_count": late_delivery_count,
                "avg_response_time_secs": avg_response_time_secs,
                "abandoned_rate_pct": abandoned_rate_pct,
                "abandoned_no_reply_pct": abandoned_no_reply_pct,
                "abandoned_disengaged_pct": abandoned_disengaged_pct,
                "abandoned_late_response_pct": abandoned_late_response_pct,
                "score1_rows": score1_with_content,
                "score2_rows": score2_with_content,
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ── Dataset-level overview metrics ────────────────────────────────────────────

def compute_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute high-level dataset statistics (after filters applied).
    """
    total_contacts = len(df)
    scored = df[df["sat_score"].notna()]
    csat_responses = len(scored)
    csat_response_rate = csat_responses / total_contacts if total_contacts else 0.0
    overall_avg_csat = float(scored["sat_score"].mean()) if csat_responses else float("nan")

    score_dist: dict[int, int] = {k: 0 for k in range(1, 6)}
    for score, cnt in scored["sat_score"].value_counts().items():
        try:
            bucket = int(round(float(score)))
            if bucket in score_dist:
                score_dist[bucket] += int(cnt)
        except (ValueError, TypeError):
            pass

    score1_rate = score_dist[1] / csat_responses if csat_responses else 0.0
    overall_csat_4_5_pct = (
        round((score_dist[4] + score_dist[5]) / csat_responses * 100, 1)
        if csat_responses else None
    )

    # Channel breakdown
    channel_stats = []
    for ch, grp in df.groupby("channel"):
        sc = grp[grp["sat_score"].notna()]
        channel_stats.append(
            {
                "channel": ch,
                "contacts": len(grp),
                "avg_csat": round(float(sc["sat_score"].mean()), 2) if len(sc) else None,
                "response_rate": round(len(sc) / len(grp) * 100, 1),
            }
        )
    channel_stats.sort(key=lambda x: x["contacts"], reverse=True)

    # Vertical breakdown
    vertical_stats = []
    for vt, grp in df.groupby("vertical_type"):
        sc = grp[grp["sat_score"].notna()]
        lc = sc[sc["sat_score"] <= 2]
        vertical_stats.append(
            {
                "vertical_type": vt if vt else "(unknown)",
                "contacts": len(grp),
                "avg_csat": round(float(sc["sat_score"].mean()), 2) if len(sc) else None,
                "low_csat_pct": round(len(lc) / len(sc) * 100, 1) if len(sc) else 0.0,
            }
        )
    vertical_stats.sort(key=lambda x: x["contacts"], reverse=True)

    return {
        "total_contacts": total_contacts,
        "csat_responses": csat_responses,
        "csat_response_rate": round(csat_response_rate, 3),
        "overall_avg_csat": round(overall_avg_csat, 2),
        "overall_csat_4_5_pct": overall_csat_4_5_pct,
        "score_distribution": score_dist,
        "score1_rate": round(score1_rate, 3),
        "channel_stats": channel_stats,
        "vertical_stats": vertical_stats,
    }


# ── Agent flagging ─────────────────────────────────────────────────────────────

def flag_agents(
    agent_df: pd.DataFrame,
    overall_avg_csat: float,
    low_csat_offset: float = 0.30,
    top_performer_offset: float = 0.30,
    min_contacts: int | None = None,
    min_csat_responses: int = 15,
) -> pd.DataFrame:
    """
    Add 'flag' column: 'PRIORITY_REVIEW', 'TOP_PERFORMER', or 'NORMAL'.
    min_contacts defaults to the dataset median if None.
    """
    df = agent_df.copy()

    if min_contacts is None:
        min_contacts = int(df["total_contacts"].median())

    low_thresh = overall_avg_csat - low_csat_offset
    top_thresh = overall_avg_csat + top_performer_offset

    volume_ok = df["total_contacts"] >= min_contacts
    responses_ok = df["csat_responses"] >= min_csat_responses

    df["flag"] = "NORMAL"
    df.loc[
        volume_ok & responses_ok & (df["avg_csat"] <= low_thresh),
        "flag",
    ] = "PRIORITY_REVIEW"
    df.loc[
        volume_ok & responses_ok & (df["avg_csat"] >= top_thresh),
        "flag",
    ] = "TOP_PERFORMER"

    df["low_csat_threshold"] = round(low_thresh, 2)
    df["top_performer_threshold"] = round(top_thresh, 2)

    return df
