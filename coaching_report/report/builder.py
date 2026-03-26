"""
Assembles all analysis outputs into a single report data model dict
that is passed directly to the Jinja2 template.
"""
from __future__ import annotations

import html as html_lib
import re
from datetime import date
from typing import Any

import pandas as pd

from analysis.clustering import CLUSTER_META
from report.llm import generate_coaching_copy

_MICRO_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+')


def _process_transcript(raw: str, max_chars: int = 800) -> tuple[str, list[str]]:
    """
    Prepare a raw transcript for display:
      1. HTML-unescape (decode &#34;, &gt;, etc.)
      2. Strip .000000 microseconds from timestamps
      3. Truncate at max_chars on a line boundary (not mid-line)
      4. HTML-escape each line exactly once (safe to render with | safe in Jinja2)
    Returns (truncated_string, list_of_escaped_lines).
    """
    text = html_lib.unescape(raw)
    text = _MICRO_RE.sub(r'\1', text)
    if len(text) > max_chars:
        chunk = text[:max_chars]
        last_nl = chunk.rfind('\n')
        text = chunk[:last_nl] if last_nl > 0 else chunk
    lines = [html_lib.escape(line) for line in text.split('\n')]
    return text, lines


# ── Conversation example selection ───────────────────────────────────────────

def select_convo_examples(
    score1_rows: pd.DataFrame,
    score2_rows: pd.DataFrame,
    max_examples: int = 3,
    max_chars: int = 800,
) -> list[dict]:
    """
    Select up to max_examples conversation examples per agent.

    Selection order:
      1. sat_score == 1 contacts, sorted most-recent first
      2. Backfill with sat_score == 2 contacts (same sort) until max_examples reached

    Each example dict contains: content, content_lines, sat_score, agent_actions,
    handling_time_mins, order_status, cancellation_reason.
    """
    def _rows_to_examples(rows: pd.DataFrame, needed: int) -> list[dict]:
        if rows.empty or needed == 0:
            return []
        # Sort most-recent first
        if "contact_creation_timestamp" in rows.columns:
            rows = rows.sort_values("contact_creation_timestamp", ascending=False)
        examples = []
        for _, row in rows.head(needed).iterrows():
            cancel = str(row.get("cancellation_reason", "")).strip()
            content_str, content_lines = _process_transcript(
                str(row.get("content", "")), max_chars
            )
            examples.append(
                {
                    "content": content_str,
                    "content_lines": content_lines,
                    "sat_score": int(round(float(row["sat_score"]))) if pd.notna(row.get("sat_score")) else None,
                    "agent_actions": str(row.get("agent_actions", "")),
                    "handling_time_mins": (
                        round(float(row["handling_time_mins"]), 1)
                        if pd.notna(row.get("handling_time_mins")) else None
                    ),
                    "order_status": str(row.get("order_status", "")).strip() or "—",
                    "cancellation_reason": cancel if cancel else "—",
                    "contact_id": str(row.get("contact_id", "")).strip() or None,
                }
            )
        return examples

    examples = _rows_to_examples(score1_rows, max_examples)
    if len(examples) < max_examples:
        examples += _rows_to_examples(score2_rows, max_examples - len(examples))
    return examples


# ── Benchmark example contact finder ─────────────────────────────────────────

def find_benchmark_example(agent_email: str, df: pd.DataFrame) -> str | None:
    """
    Return the contact_id of the best example conversation for a benchmark agent.
    Prefers sat_score == 5; falls back to sat_score == 4.
    Within the chosen score tier, takes the contact with the highest num_messages.
    Requires content to be non-null and longer than 100 characters.
    Returns None if no qualifying contact exists.
    """
    agent_df = df[df["agent_email"] == agent_email]
    qualifying = agent_df[
        agent_df["content"].notna() & (agent_df["content"].str.len() > 100)
    ]
    for target_score in [5, 4]:
        tier = qualifying[
            qualifying["sat_score"].apply(
                lambda x: pd.notna(x) and int(round(float(x))) == target_score
            )
        ]
        if not tier.empty:
            best_row = tier.loc[tier["num_messages"].idxmax()]
            cid = str(best_row.get("contact_id", "")).strip()
            return cid if cid else None
    return None


# ── Top performer context string (used in LLM prompt) ────────────────────────

def _top_performer_context(top_performers: list[dict]) -> str:
    if not top_performers:
        return "No top performers identified in this dataset."
    best = top_performers[0]
    ar = best.get("actioned_rate_pct", 0)
    comp_rate = best.get("comp_count", 0) / best["total_contacts"] if best["total_contacts"] else 0
    return (
        f"{best['agent_email']} — {best['csat_4_5_pct']}% 4-5 scores "
        f"(avg CSAT {best['avg_csat']}/5), "
        f"{ar:.1f}% actioned, calls on {best['call_rate_pct']}% of contacts, "
        f"compensation on {comp_rate:.1%} of contacts. "
        f"Handles {best['total_contacts']} contacts at {best['avg_handling_time_mins']} min avg HT."
    )


def _cluster_benchmark(
    cluster_letter: str,
    top_performers: list[dict],
    team_avg_actioned: float = 0.0,
    team_avg_4_5_pct: float | None = None,
    contact_df: pd.DataFrame | None = None,
) -> dict:
    """
    Select the most relevant top performer for a cluster and generate a tailored narrative.
    Falls back to team averages if no top performers exist.
    Selection rules (all from the actual TOP_PERFORMER pool; never hardcoded):
      A → highest actioned_rate_pct
      B → highest actioned_rate_pct + call_rate_pct combined
      C → highest csat_4_5_pct among TPs with call_rate_pct >= 50 AND comp_rate_pct >= 15;
          fallback: highest csat_4_5_pct overall
      D → highest csat_4_5_pct overall
    """
    if not top_performers:
        # Fallback: team averages
        return {
            "tp_handle": None,
            "tp_csat_4_5_pct": team_avg_4_5_pct,
            "narrative": (
                f"Team average: {team_avg_4_5_pct}% 4–5 scores, "
                f"{team_avg_actioned:.1f}% actioned rate. "
                f"Top performers on this contact type combine calling with compensation "
                f"and stay under 13 min handling time."
            ),
            "benchmark_contact_id": None,
        }

    if cluster_letter == "A":
        best = max(top_performers, key=lambda x: x.get("actioned_rate_pct", 0))
        narrative = (
            f"{best['agent_email'].split('@')[0]} actioned {best['actioned_rate_pct']}% of contacts "
            f"and scored {best['csat_4_5_pct']}% 4–5. "
            f"For 'Check Order Status', taking action (call + compensation) on every delay is the baseline — "
            f"not a bonus."
        )
    elif cluster_letter == "B":
        best = max(
            top_performers,
            key=lambda x: x.get("actioned_rate_pct", 0) + x.get("call_rate_pct", 0),
        )
        narrative = (
            f"{best['agent_email'].split('@')[0]} actioned {best['actioned_rate_pct']}% of contacts "
            f"and called on {best['call_rate_pct']}% ({best['csat_4_5_pct']}% 4–5 scores). "
            f"Using a fuller toolkit — calls, compensation, and escalations — "
            f"is the single highest-impact lever for this cluster."
        )
    elif cluster_letter == "C":
        qualified = [
            x for x in top_performers
            if x.get("call_rate_pct", 0) >= 50 and x.get("comp_rate_pct", 0) >= 15
        ]
        pool = qualified if qualified else top_performers
        best = max(pool, key=lambda x: x.get("csat_4_5_pct") or 0)
        narrative = (
            f"{best['agent_email'].split('@')[0]} combines call ({best['call_rate_pct']}%) + "
            f"compensation ({best['comp_rate_pct']}%) and achieves {best['csat_4_5_pct']}% 4–5 scores. "
            f"Action quality matters: the right action at the right moment lifts CSAT "
            f"even when total action count looks similar."
        )
    else:  # D — active agents where execution quality is the lever
        best = max(top_performers, key=lambda x: x.get("csat_4_5_pct") or 0)
        narrative = (
            f"{best['agent_email'].split('@')[0]} scores {best['csat_4_5_pct']}% 4–5 "
            f"in {best['avg_handling_time_mins']} min avg. "
            f"With an already-high action rate, quality of execution is the lever — "
            f"concise resolution and picking the right action first time."
        )

    benchmark_contact_id = (
        find_benchmark_example(best["agent_email"], contact_df)
        if contact_df is not None else None
    )
    return {
        "tp_handle": best["agent_email"].split("@")[0],
        "tp_csat_4_5_pct": best["csat_4_5_pct"],
        "narrative": narrative,
        "benchmark_contact_id": benchmark_contact_id,
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_report(
    overview: dict[str, Any],
    agent_df: pd.DataFrame,
    drivers: dict[str, Any],
    market: str,
    contact_reason: str,
    run_date: date | None = None,
    use_llm: bool = True,
    llm_model: str = "claude-sonnet-4-6",
    llm_max_tokens: int = 1200,
    max_convo_examples: int = 3,
    convo_max_chars: int = 800,
    contact_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Build the full report data model.
    Returns a dict ready to be passed to the Jinja2 template.
    """
    if run_date is None:
        run_date = date.today()

    # ── Split agents by flag ──────────────────────────────────────────────────
    pr_df = agent_df[agent_df["flag"] == "PRIORITY_REVIEW"].copy()
    tp_df = agent_df[agent_df["flag"] == "TOP_PERFORMER"].copy()

    team_avg_actioned = float(agent_df["actioned_rate_pct"].mean()) if not agent_df.empty else 53.0
    # Team-wide csat_4_5_pct average across all agents with responses
    valid_pcts = agent_df["csat_4_5_pct"].dropna()
    team_avg_4_5_pct = round(float(valid_pcts.mean()), 1) if not valid_pcts.empty else None

    valid_abandoned = agent_df["abandoned_rate_pct"].dropna() if "abandoned_rate_pct" in agent_df.columns else pd.Series(dtype=float)
    team_avg_abandoned_rate = round(float(valid_abandoned.mean()), 1) if not valid_abandoned.empty else 0.0

    # ── Build top performer list (top 5 by csat_4_5_pct) ────────────────────
    top_performers = []
    for _, row in tp_df.sort_values("csat_4_5_pct", ascending=False, na_position="last").head(5).iterrows():
        comp_rate = row["comp_count"] / row["total_contacts"] if row["total_contacts"] else 0
        call_rate = row.get("call_contact_count", row["call_count"]) / row["total_contacts"] if row["total_contacts"] else 0
        vendor_call_rate = row["vendor_call_count"] / row["total_contacts"] if row["total_contacts"] else 0
        top_actions_formatted = ", ".join(
            f"{k} ×{v}" for k, v in list((row.get("top_actions") or {}).items())[:4]
        )
        top_performers.append(
            {
                "agent_email": row["agent_email"],
                "avg_csat": row["avg_csat"],
                "csat_4_5_pct": row.get("csat_4_5_pct"),
                "total_contacts": row["total_contacts"],
                "actioned_rate_pct": row["actioned_rate_pct"],
                "low_csat_pct": row["low_csat_pct"],
                "avg_handling_time_mins": row["avg_handling_time_mins"],
                "comp_rate_pct": round(comp_rate * 100, 1),
                "call_rate_pct": round(call_rate * 100, 1),
                "vendor_call_rate_pct": round(vendor_call_rate * 100, 1),
                "comp_count": row.get("comp_count", 0),
                "call_count": row.get("call_count", 0),
                "top_actions_formatted": top_actions_formatted,
            }
        )

    tp_context = _top_performer_context(top_performers)
    best_tp = top_performers[0] if top_performers else None
    best_tp_4_5_pct = best_tp["csat_4_5_pct"] if best_tp else None
    best_tp_handle = best_tp["agent_email"].split("@")[0] if best_tp else "—"

    # ── Build priority review agent cards ────────────────────────────────────
    priority_agents: list[dict] = []

    for _, row in pr_df.sort_values(
        ["cluster", "avg_csat"], ascending=[True, True]
    ).iterrows():
        agent_data: dict[str, Any] = {
            "agent_email": row["agent_email"],
            "agent_name": row.get("agent_name") or None,
            "cluster": row["cluster"],
            "cluster_label": CLUSTER_META.get(row["cluster"], {}).get("label", ""),
            "avg_csat": row["avg_csat"],
            "csat_4_5_pct": row.get("csat_4_5_pct"),
            "actioned_rate_pct": row["actioned_rate_pct"],
            "low_csat_pct": row["low_csat_pct"],
            "avg_handling_time_mins": row["avg_handling_time_mins"],
            "avg_response_time_secs": row.get("avg_response_time_secs"),
            "abandoned_rate_pct": row.get("abandoned_rate_pct", 0.0),
            "abandoned_no_reply_pct": row.get("abandoned_no_reply_pct", 0.0),
            "abandoned_disengaged_pct": row.get("abandoned_disengaged_pct", 0.0),
            "abandoned_late_response_pct": row.get("abandoned_late_response_pct", 0.0),
            "team_avg_abandoned_rate": team_avg_abandoned_rate,
            "total_contacts": row["total_contacts"],
            "csat_responses": row["csat_responses"],
            "score_distribution": row["score_distribution"],
            "top_actions": row.get("top_actions") or {},
            "action_counter": row.get("action_counter") or {},
            "comp_count": row["comp_count"],
            "call_count": row["call_count"],
            "vendor_call_count": row["vendor_call_count"],
            "cancel_count": row["cancel_count"],
            "reassign_count": row["reassign_count"],
            "dominant_action": row["dominant_action"],
            "dominant_action_share": row["dominant_action_share"],
            "low_csat_order_statuses": row["low_csat_order_statuses"],
            "low_csat_cancel_reasons": row["low_csat_cancel_reasons"],
            "low_csat_delivered_pct": row["low_csat_delivered_pct"],
            "late_delivery_count": row["late_delivery_count"],
            "low_csat_threshold": row.get("low_csat_threshold"),
            "team_avg_actioned_rate": team_avg_actioned,
            "team_avg_4_5_pct": team_avg_4_5_pct,
        }

        # Conversation examples (3, with score-2 backfill)
        score1_rows = row.get("score1_rows")
        score2_rows = row.get("score2_rows")
        convo_examples = select_convo_examples(
            score1_rows if score1_rows is not None else pd.DataFrame(),
            score2_rows if score2_rows is not None else pd.DataFrame(),
            max_examples=max_convo_examples,
            max_chars=convo_max_chars,
        )
        agent_data["convo_example_text"] = convo_examples[0]["content"] if convo_examples else ""

        # Generate coaching copy
        print(f"  → Generating coaching copy for {row['agent_email']} (cluster {row['cluster']})…")
        coaching = generate_coaching_copy(
            agent_data,
            top_performer_context=tp_context,
            use_llm=use_llm,
            model=llm_model,
            max_tokens=llm_max_tokens,
        )

        agent_data["coaching"] = coaching
        agent_data["convo_examples"] = convo_examples

        priority_agents.append(agent_data)

    # Flag agents in top 25% of abandoned_rate_pct (callout in diagnosis block)
    ab_rates = [a.get("abandoned_rate_pct", 0.0) for a in priority_agents]
    if len(ab_rates) >= 4:
        ab_p75 = float(pd.Series(ab_rates).quantile(0.75))
        for a in priority_agents:
            a["abandoned_high"] = a.get("abandoned_rate_pct", 0.0) >= ab_p75
    else:
        for a in priority_agents:
            a["abandoned_high"] = a.get("abandoned_rate_pct", 0.0) > 0.0

    # ── Group priority agents by cluster ─────────────────────────────────────
    clusters: dict[str, dict] = {}
    for letter, meta in CLUSTER_META.items():
        agents_in_cluster = [a for a in priority_agents if a["cluster"] == letter]
        clusters[letter] = {
            **meta,
            "agents": agents_in_cluster,
            "agent_handles": [a["agent_email"].split("@")[0] for a in agents_in_cluster],
            "benchmark": (
                _cluster_benchmark(
                    letter, top_performers,
                    team_avg_actioned=team_avg_actioned,
                    team_avg_4_5_pct=team_avg_4_5_pct,
                    contact_df=contact_df,
                ) if agents_in_cluster else None
            ),
        }

    # ── Final report model ────────────────────────────────────────────────────
    return {
        "market": market.upper(),
        "contact_reason": contact_reason,
        "run_date": run_date.strftime("%B %d, %Y"),
        "run_date_slug": run_date.strftime("%Y%m%d"),
        # Overview
        "total_contacts": overview["total_contacts"],
        "csat_responses": overview["csat_responses"],
        "csat_response_rate": overview["csat_response_rate"],
        "overall_avg_csat": overview["overall_avg_csat"],
        "overall_csat_4_5_pct": overview.get("overall_csat_4_5_pct"),
        "score_distribution": overview["score_distribution"],
        "score1_rate": overview["score1_rate"],
        "score1_rate_pct": round(overview["score1_rate"] * 100, 1),
        "agents_flagged": len(priority_agents),
        "channel_stats": overview["channel_stats"],
        "vertical_stats": overview["vertical_stats"],
        # Thresholds
        "low_csat_threshold": round(overview["overall_avg_csat"] - 0.30, 2),
        "target_csat": round(overview["overall_avg_csat"] + 0.30, 2),
        # Agent sections
        "priority_agents": priority_agents,
        "top_performers": top_performers,
        "best_tp_4_5_pct": best_tp_4_5_pct,
        "best_tp_handle": best_tp_handle,
        # Clusters
        "clusters": clusters,
        "cluster_order": ["A", "B", "C", "D"],
        # Drivers
        "drivers": drivers,
        "team_avg_abandoned_rate": team_avg_abandoned_rate,
    }
