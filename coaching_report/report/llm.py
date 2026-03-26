"""
LLM-assisted coaching copy generation via the Anthropic API.

Falls back to rule-based template text if:
  • ANTHROPIC_API_KEY is not set
  • use_llm is False in config
  • An API error occurs
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any

# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_signals(a: dict[str, Any]) -> list[str]:
    """Return the list of coaching signal strings triggered for this agent."""
    signals = []
    ar = a["actioned_rate_pct"]
    comp_rate = a["comp_count"] / a["total_contacts"] if a["total_contacts"] else 0
    call_rate = a["call_count"] / a["total_contacts"] if a["total_contacts"] else 0
    ht = a["avg_handling_time_mins"]
    dom_share = a["dominant_action_share"]
    lc_del = a["low_csat_delivered_pct"]
    ld_count = a["late_delivery_count"]

    if ar < 20:
        signals.append(
            f"NEAR_ZERO_ACTION: Only {ar:.1f}% of contacts actioned. Agent is almost "
            "never taking action — walk through the full action toolkit in the 1:1."
        )
    elif ar < 40:
        signals.append(
            f"LOW_ACTION_RATE: Only {ar:.1f}% actioned. Ask the agent what their "
            "internal threshold is for deciding to act, then lower it."
        )

    if comp_rate < 0.08:
        signals.append(
            f"COMP_UNDERUSED: Only {a['comp_count']} compensation actions on "
            f"{a['total_contacts']} contacts ({comp_rate:.1%}). Every delayed order "
            "requires at minimum a compensation consideration."
        )

    if call_rate < 0.10:
        signals.append(
            f"CALL_AVOIDANCE: Only {a['call_count']} calls on {a['total_contacts']} "
            f"contacts ({call_rate:.1%}). On orders delayed > 30 min, calling the "
            "customer or vendor should be the first intervention."
        )

    if ht > 13:
        signals.append(
            f"LONG_HANDLE_TIME: Avg {ht:.1f} min — above the 13-min threshold. "
            "Coach a structured 7-minute resolution model: greet → investigate "
            "(2 min) → action → confirm → close."
        )

    if dom_share > 0.60 and a["dominant_action"]:
        label = a["dominant_action"].replace("has_", "").replace("_done", "").replace("_", " ")
        signals.append(
            f"SINGLE_ACTION_RELIANCE: '{label}' makes up {dom_share:.1%} of all "
            "actions taken. Walk through the full toolkit and practice "
            "combined-action scenarios."
        )

    if lc_del > 0.40:
        signals.append(
            f"DELIVERED_LOW_CSAT: {lc_del:.1%} of low-CSAT contacts are DELIVERED "
            "orders — arrived but still scored 1–2. Proactive compensation needed "
            "even post-delivery on long delays."
        )

    if ld_count >= 5:
        signals.append(
            f"LATE_DELIVERY_PATTERN: {ld_count} LATE_DELIVERY contacts in low-CSAT. "
            "Every LATE_DELIVERY contact must have compensation applied — no exceptions."
        )

    rt = a.get("avg_response_time_secs")
    if rt is not None and rt > 90:
        signals.append(
            f"SLOW_RESPONSE_TIME: Agent takes {rt:.0f}s on average to respond to customer "
            "messages. Review for concurrent chat overload or disengagement. "
            "Top performers average under 45 seconds per response turn."
        )

    ab_no_reply = a.get("abandoned_no_reply_pct", 0) or 0
    ab_disengaged = a.get("abandoned_disengaged_pct", 0) or 0
    ab_late = a.get("abandoned_late_response_pct", 0) or 0

    if ab_no_reply > 10:
        signals.append(
            f"ABANDONED_NO_REPLY: {ab_no_reply:.1f}% of contacts end with an unanswered "
            "customer message. Check for concurrent chat overload or contacts being closed "
            "without reading the final customer message."
        )

    if ab_disengaged > 15:
        signals.append(
            f"ABANDONED_DISENGAGED: Customers disengage after the agent's last message on "
            f"{ab_disengaged:.1f}% of contacts without a proper closure. Review whether "
            "responses are arriving too late, feel dismissive, or fail to offer a next step."
        )

    if ab_late > 10:
        signals.append(
            f"ABANDONED_LATE_RESPONSE: {ab_late:.1f}% of contacts have a slow agent response "
            "(>5 min) that caused the customer to disengage. Investigate concurrent chat load "
            "during these contacts."
        )

    if not signals:
        signals.append(
            "GENERAL_QUALITY_GAP: Metrics don't match a single pattern — "
            "likely a communication quality issue. Review call recordings together."
        )

    return signals


def _rule_based_copy(a: dict[str, Any]) -> dict[str, Any]:
    """
    Generate coaching copy using rule-based templates.
    Returns {'diagnosis': str, 'coaching_actions': list[dict]}.
    """
    ar = a["actioned_rate_pct"]
    comp_rate = a["comp_count"] / a["total_contacts"] if a["total_contacts"] else 0
    call_rate = a["call_count"] / a["total_contacts"] if a["total_contacts"] else 0
    ht = a["avg_handling_time_mins"]
    dom_share = a["dominant_action_share"]
    dom_label = (
        a["dominant_action"].replace("has_", "").replace("_done", "").replace("_", " ")
        if a["dominant_action"] else "single action"
    )
    team_avg = a.get("team_avg_actioned_rate", 53.0)

    # Diagnosis
    if ar < 20:
        diagnosis = (
            f"Only {a['comp_count'] + a['call_count'] + a['cancel_count'] + a['reassign_count']} "
            f"actions taken across {a['total_contacts']} contacts. "
            f"In {round(a['total_contacts'] * (1 - ar/100))} out of {a['total_contacts']} contacts, "
            "no action was recorded. Agent appears to diagnose the issue, offer apologies, "
            "and close the chat without a real resolution attempt."
        )
    elif ar < 50:
        diagnosis = (
            f"{round(100 - ar, 1)}% of contacts have no action. "
            f"When acting, the agent primarily uses {dom_label} "
            f"({dom_share:.1%} of actions). The callback offer is used but often "
            "the call doesn't complete — chat closes after a single failed attempt. "
            "Compensation is severely underused."
        )
    elif ar < 70:
        diagnosis = (
            f"Acting on {ar:.1f}% of contacts, but the action mix is skewed. "
            f"The agent over-relies on '{dom_label}' ({dom_share:.1%} of actions). "
            f"Compensation rate is {comp_rate:.1%} — below the top performer standard. "
            "Many contacts run long without reaching a clear resolution."
        )
    else:
        diagnosis = (
            f"High action rate ({ar:.1f}%) but CSAT remains low. "
            f"The agent calls and compensates, but conversations average "
            f"{ht:.1f} minutes without proportionate improvement in CSAT. "
            "This is a communication quality gap — the right actions are taken "
            "but not delivered in a way that resonates with customers."
        )

    # Coaching actions
    actions = []
    target_rate = min(round(ar + 20, 0), 70.0)
    current_4_5 = a.get("csat_4_5_pct") or 0.0
    team_4_5 = a.get("team_avg_4_5_pct") or (current_4_5 + 10)
    target_4_5 = round(min(current_4_5 + 10, team_4_5), 1)

    if ar < 20:
        actions.append({
            "headline": "Walk through the action toolkit with a live example",
            "body": (
                f"Show the agent the actions available (compensation, reassign, vendor call, "
                f"customer call, confirm cancel) and ask: 'For a delayed order — which "
                f"actions should you be offering within the first 5 minutes?' Every contact "
                f"where the order is late must result in at least one action — no exceptions."
            ),
        })
    elif ar < 40:
        actions.append({
            "headline": "Identify the internal decision threshold and lower it",
            "body": (
                f"Ask the agent: 'What needs to be true before you take an action on a "
                f"contact?' Walk through 5 non-actioned contacts and challenge each "
                f"decision. If the order is delayed > 20 min beyond ETA, action is "
                f"mandatory — not optional."
            ),
        })

    if comp_rate < 0.08:
        actions.append({
            "headline": f"Address severely underused compensation ({a['comp_count']} uses in {a['total_contacts']} contacts)",
            "body": (
                f"Only {a['comp_count']} compensation actions across {a['total_contacts']} contacts "
                f"({comp_rate:.1%}) — far below the top performer rate (~20%). "
                "Prompt the agent: for any order delayed > 30 minutes, compensation must be "
                "the minimum default consideration. Review the last 5 non-compensated delay contacts together."
            ),
        })

    if call_rate < 0.10:
        actions.append({
            "headline": "Introduce calling as a mandatory tool for live delayed orders",
            "body": (
                f"Only {a['call_count']} calls across {a['total_contacts']} contacts "
                f"({call_rate:.1%}). Top performers call on 55–73% of contacts. "
                "On orders delayed over 30 minutes, calling the customer demonstrates "
                "urgency and care that chat alone cannot replicate. Practice the "
                "opening phrase for an outbound call in the 1:1."
            ),
        })

    if ht > 13:
        actions.append({
            "headline": f"Coach a structured resolution model to reduce {ht:.1f}-min avg handle time",
            "body": (
                "Long contacts do not produce better CSAT — they produce frustrated customers. "
                "Coach the 7-minute model: greet → investigate (2 min) → apply action → "
                "confirm → close. If a contact exceeds 12 messages without a resolution, "
                "the agent must escalate or apply a concrete action rather than continue investigating in chat."
            ),
        })

    if dom_share > 0.60:
        actions.append({
            "headline": f"Break the over-reliance on '{dom_label}'",
            "body": (
                f"'{dom_label}' makes up {dom_share:.1%} of all actions — "
                "a single-tool pattern that fails when that tool isn't appropriate. "
                "Walk through the full action toolkit and practice a combined-action "
                "scenario: call the customer + call the vendor + apply compensation."
            ),
        })

    if a["low_csat_delivered_pct"] > 0.40:
        actions.append({
            "headline": f"Focus on DELIVERED + low-CSAT contacts ({round(a['low_csat_delivered_pct']*100)}% of low-CSAT)",
            "body": (
                "Orders arriving late but still delivered are scoring 1–2. "
                "Arriving late is still a failure — always compensate, even post-delivery. "
                "Pull these contacts and walk through each one: what action was (or wasn't) taken?"
            ),
        })

    if a["late_delivery_count"] >= 5:
        actions.append({
            "headline": f"Create a rule: LATE_DELIVERY = compensation, no exceptions ({a['late_delivery_count']} contacts)",
            "body": (
                f"{a['late_delivery_count']} LATE_DELIVERY contacts in the low-CSAT set. "
                "Late delivery is the highest-damage cancellation reason in the dataset. "
                "Walk through these contacts and confirm: was compensation applied? "
                "If not, establish a hard rule — LATE_DELIVERY always triggers compensation."
            ),
        })

    # Fallback actions to pad to exactly 3 when fewer signals fire
    fallback_actions = [
        {
            "headline": "Review Score-1 contacts together in the 1:1",
            "body": (
                f"Pull {min(5, a['csat_responses'])} Score-1 contacts and walk through them "
                "with the agent. For each one ask: 'What did the customer expect? What did we "
                "deliver? What one thing would have changed the outcome?' Build the habit of "
                "self-diagnosis before the next shift."
            ),
        },
        {
            "headline": "Pair with a top performer for call shadowing",
            "body": (
                "Arrange a shadowing session where this agent listens to how a top performer "
                "opens, investigates, and closes a delayed-order contact. Focus specifically "
                "on: how quickly they move to action, how they frame compensation, and how "
                "they confirm resolution before ending the chat."
            ),
        },
        {
            "headline": "Set a weekly CSAT target and review mid-week",
            "body": (
                f"Current avg CSAT: {a['avg_csat']}. Set a concrete target for this week "
                f"and schedule a 10-minute mid-week check-in to review progress. "
                "Visibility alone often drives improvement — the agent needs to know "
                "their number is being watched daily, not just at the end of the week."
            ),
        },
    ]

    for fb in fallback_actions:
        if len(actions) >= 3:
            break
        actions.append(fb)

    # Cap at 3 and ensure measurable target is appended to the last action
    actions = actions[:3]
    actions[-1]["body"] += (
        f" Target: raise % of 4–5 scores from {current_4_5:.1f}% to {target_4_5:.1f}% "
        f"(team average: {team_4_5:.1f}%) within 2 weeks."
    )

    return {"diagnosis": diagnosis, "coaching_actions": actions}


# ── LLM-assisted generation ───────────────────────────────────────────────────

def _build_prompt(a: dict[str, Any], signals: list[str], top_performer_context: str) -> str:
    top_actions_str = "\n".join(
        f"  {k}: {v}" for k, v in (a.get("top_actions") or {}).items()
    ) or "  (none recorded)"

    convo = a.get("convo_example_text") or "(no Score-1 conversation available)"
    convo_truncated = convo[:800] if len(convo) > 800 else convo

    cancel_reasons = a.get("low_csat_cancel_reasons") or {}
    order_statuses = a.get("low_csat_order_statuses") or {}

    team_avg = a.get("team_avg_actioned_rate", 53.0)
    comp_rate = a["comp_count"] / a["total_contacts"] if a["total_contacts"] else 0
    call_rate = a["call_count"] / a["total_contacts"] if a["total_contacts"] else 0

    return f"""You are a customer service quality coach. Generate a data-driven coaching report for a Team Leader's 1:1 session tomorrow.

Return ONLY valid JSON with this exact structure:
{{
  "diagnosis": "2-3 sentence root cause paragraph explaining WHY this agent's CSAT is low, referencing specific numbers",
  "coaching_actions": [
    {{"headline": "Concise action title (bold instruction)", "body": "Detailed what/why/how — include specific numbers and one measurable target"}},
    {{"headline": "...", "body": "..."}},
    {{"headline": "...", "body": "..."}}
  ]
}}

AGENT: {a['agent_email']}
CLUSTER: {a.get('cluster', '?')} — {a.get('cluster_label', '')}

KEY METRICS:
  4–5 Score %: {a.get('csat_4_5_pct')}%  ← PRIMARY DISPLAY METRIC
  Avg CSAT (internal): {a['avg_csat']} / 5  (low threshold: {a.get('low_csat_threshold', '?')})
  Total contacts: {a['total_contacts']}
  CSAT responses: {a['csat_responses']}
  Low CSAT %: {a['low_csat_pct']}%
  Actioned rate: {a['actioned_rate_pct']}%  (team avg: {team_avg:.0f}%)
  Avg handling time: {a['avg_handling_time_mins']} min
  Avg response time (chat): {f"{a['avg_response_time_secs']}s" if a.get('avg_response_time_secs') is not None else "n/a (no chat contacts)"}
  Abandoned rate: {a.get('abandoned_rate_pct', 0)}% (no-reply: {a.get('abandoned_no_reply_pct', 0)}%, disengaged: {a.get('abandoned_disengaged_pct', 0)}%, late-response: {a.get('abandoned_late_response_pct', 0)}%)
  Score distribution: {a['score_distribution']}

ACTION BREAKDOWN:
{top_actions_str}

  Compensation: {a['comp_count']} uses ({comp_rate:.1%} of contacts)
  Calls (outbound + customer): {a['call_count']} ({call_rate:.1%} of contacts)
  Vendor/courier calls: {a['vendor_call_count']}
  Cancellations: {a['cancel_count']}
  Reassignments: {a['reassign_count']}
  Dominant action: '{a['dominant_action']}' ({a['dominant_action_share']:.1%} of all action tokens)

LOW-CSAT CONTEXT:
  Top order statuses: {order_statuses}
  Top cancellation reasons: {cancel_reasons}
  % of low-CSAT that are DELIVERED: {a['low_csat_delivered_pct']:.1%}
  LATE_DELIVERY count in low-CSAT: {a['late_delivery_count']}

COACHING SIGNALS TRIGGERED:
{chr(10).join(f"  • {s}" for s in signals)}

SCORE-1 CONVERSATION EXAMPLE:
{convo_truncated}

TOP PERFORMER BENCHMARK:
{top_performer_context}

INSTRUCTIONS:
- Diagnosis: 2-3 sentences, cite specific numbers, explain the root cause clearly
- Each coaching action: direct, actionable, TL can use in a 1:1 tomorrow
- At least one action must include a specific measurable target referencing csat_4_5_pct (e.g. "raise % of 4–5 scores from X% to Y% within 2 weeks")
- Tone: professional, supportive, data-driven — not punitive
- Do NOT add any text outside the JSON object"""


def generate_coaching_copy(
    agent_data: dict[str, Any],
    top_performer_context: str,
    use_llm: bool = True,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """
    Generate coaching diagnosis + 3 actions for a single agent.

    Returns {'diagnosis': str, 'coaching_actions': list[{headline, body}]}.
    Falls back to rule-based copy on any error.
    """
    signals = _rule_based_signals(agent_data)

    if not use_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        return _rule_based_copy(agent_data)

    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = _build_prompt(agent_data, signals, top_performer_context)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        # Validate structure
        assert "diagnosis" in result and "coaching_actions" in result
        assert len(result["coaching_actions"]) == 3
        return result

    except Exception as exc:
        print(f"  ⚠  LLM coaching copy failed for {agent_data.get('agent_email', '?')}: {exc}")
        print("     Falling back to rule-based copy.")
        return _rule_based_copy(agent_data)
