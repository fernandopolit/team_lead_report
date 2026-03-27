#!/usr/bin/env python3
"""
TL Coaching Report — entry point.

Usage:
  python main.py                                    # use config.yaml defaults (CSV)
  python main.py --source sheets                    # read from Google Sheets
  python main.py --market MX --contact-reason "check order status"
  python main.py --file path/to/data.csv
  python main.py --start 2026-03-01 --end 2026-03-26
  python main.py --no-llm                           # rule-based coaching copy only

Environment variables (override config.yaml; used in Cloud Run):
  GOOGLE_SHEET_ID         Google Sheet ID containing the data
  GOOGLE_SHEET_TAB        Worksheet tab name (default: Sheet1)
  DRIVE_FOLDER_ID         Google Drive folder ID for report delivery
  DRIVE_FILENAME_MODE     'overwrite' (default) or 'dated'
  REPORT_MARKET           Market filter, e.g. ES or MX
  REPORT_CONTACT_REASON   Contact reason substring filter (optional)
  REPORT_TIMEZONE         Timezone for run date (default: Europe/Madrid)
  GCP_PROJECT_ID          GCP project ID (informational / logging)
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import click
import yaml

# Make the project root importable when running as `python main.py`
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from data.loader import load_csv, load_latest, load_from_sheets
from analysis.filters import apply_filters
from analysis.metrics import compute_agent_metrics, compute_dataset_overview, flag_agents
from analysis.clustering import add_clusters
from analysis.drivers import compute_drivers
from report.builder import build_report


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def render_html(report_data: dict, template_path: Path) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(template_path.name)
    return template.render(**report_data)


def save_report(html: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


@click.command()
@click.option("--config", default="config.yaml", show_default=True,
              help="Path to config.yaml")
@click.option("--source", default="csv", type=click.Choice(["csv", "sheets"]),
              show_default=True, help="Data source: local CSV or Google Sheets")
@click.option("--market", default=None,
              help="Override country_code filter (e.g. ES, MX)")
@click.option("--contact-reason", default=None,
              help="Override contact_reason_contains filter")
@click.option("--file", "csv_file", default=None, type=click.Path(),
              help="Use a specific CSV file instead of the latest in csv_dir")
@click.option("--start", default=None,
              help="Start date filter YYYY-MM-DD (optional)")
@click.option("--end", default=None,
              help="End date filter YYYY-MM-DD (optional)")
@click.option("--no-llm", is_flag=True, default=False,
              help="Disable LLM coaching copy (use rule-based fallback)")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the generated report in the default browser")
def main(
    config: str,
    source: str,
    market: str | None,
    contact_reason: str | None,
    csv_file: str | None,
    start: str | None,
    end: str | None,
    no_llm: bool,
    open_browser: bool,
) -> None:
    try:
        _run(config, source, market, contact_reason, csv_file, start, end,
             no_llm, open_browser)
    except Exception as exc:
        click.echo(f"\n❌ Fatal error: {exc}", err=True)
        sys.exit(1)


def _run(
    config: str,
    source: str,
    market: str | None,
    contact_reason: str | None,
    csv_file: str | None,
    start: str | None,
    end: str | None,
    no_llm: bool,
    open_browser: bool,
) -> None:
    config_path = (ROOT / config).resolve()
    cfg = load_config(config_path)

    # ── Resolve settings: CLI > env vars > config.yaml ───────────────────────
    # Market is resolved after loading data so it can be auto-detected if unset
    _market_override = (
        market
        or os.environ.get("REPORT_MARKET")
        or cfg["filters"].get("country_code")
        or None
    )
    # Treat empty string or literal "None" as no override (auto-detect from data)
    if _market_override and _market_override.strip().lower() in ("none", ""):
        _market_override = None

    _raw_reason = (
        contact_reason
        or os.environ.get("REPORT_CONTACT_REASON")
        or cfg["filters"].get("contact_reason_contains")
    )
    # Treat empty string or the literal string "None" (common GitHub var mistake) as no filter
    reason = _raw_reason if _raw_reason and _raw_reason.strip().lower() not in ("none", "") else None

    date_start = start or cfg["date_range"].get("start")
    date_end = end or cfg["date_range"].get("end")

    use_llm = not no_llm and cfg["coaching"].get("use_llm", True)
    llm_model = cfg["coaching"].get("llm_model", "claude-sonnet-4-6")
    llm_max_tokens = cfg["coaching"].get("llm_max_tokens", 1200)
    max_convo = cfg["coaching"].get("max_convo_examples", 2)
    convo_chars = cfg["coaching"].get("convo_max_chars", 800)
    closing_phrases = cfg.get("abandonment", {}).get("closing_phrases", None)

    ds_cfg = cfg["data_source"]
    thr = cfg["thresholds"]

    # ── GCP / Drive config (env vars only — never in config.yaml) ────────────
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    sheet_tab = os.environ.get("GOOGLE_SHEET_TAB", "Sheet1")
    drive_folder_id = os.environ.get("DRIVE_FOLDER_ID")
    drive_mode = os.environ.get("DRIVE_FILENAME_MODE", "overwrite")

    # If GOOGLE_SHEET_ID is set in the environment, treat as sheets mode
    if sheet_id and source == "csv":
        source = "sheets"

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo("📂 Loading data…")
    if source == "sheets":
        if not sheet_id:
            raise ValueError(
                "GOOGLE_SHEET_ID environment variable is required for --source sheets"
            )
        click.echo(f"   Source: Google Sheet {sheet_id} / tab '{sheet_tab}'")
        df = load_from_sheets(sheet_id, sheet_tab)
        src_label = f"sheets:{sheet_id}"
    elif csv_file:
        df = load_csv(csv_file)
        src_label = Path(csv_file).name
        click.echo(f"   Source: {src_label}")
    else:
        csv_dir = ds_cfg["csv_dir"]
        df, src_path = load_latest(csv_dir, base_dir=str(ROOT))
        src_label = src_path.name
        click.echo(f"   Latest CSV: {src_label}")

    click.echo(f"   Rows loaded: {len(df):,}")

    # ── Auto-detect market from data if not explicitly set ────────────────────
    if _market_override:
        market = _market_override.upper()
    else:
        country_counts = df["country_code"].str.upper().value_counts()
        if len(country_counts) == 1:
            market = country_counts.index[0]
            click.echo(f"   Auto-detected market: {market}")
        else:
            market = country_counts.index[0]
            click.echo(
                f"   Multiple markets in data: {list(country_counts.index)} "
                f"— using most common: {market}"
            )

    # ── Filter ────────────────────────────────────────────────────────────────
    reason_label = f"'{reason}'" if reason else "all contact reasons"
    click.echo(f"🔍 Filtering → market={market}, contact_reason={reason_label}")
    filtered = apply_filters(df, country_code=market, contact_reason_contains=reason,
                             date_start=date_start, date_end=date_end)
    click.echo(f"   Rows after filter: {len(filtered):,}")

    # ── Auto-detect contact reason label from data ────────────────────────────
    if not reason:
        unique_reasons = filtered["contact_reason_l4"].value_counts()
        n = len(unique_reasons)
        if n == 1:
            contact_reason_label = unique_reasons.index[0].title()
        elif n <= 3:
            contact_reason_label = " / ".join(r.title() for r in unique_reasons.index)
        else:
            top = unique_reasons.index[0].title()
            contact_reason_label = f"{top} (+{n - 1} more)"
    else:
        contact_reason_label = reason.title()

    # ── Analysis ──────────────────────────────────────────────────────────────
    click.echo("📊 Computing metrics…")
    overview = compute_dataset_overview(filtered)
    click.echo(
        f"   Contacts: {overview['total_contacts']:,} | "
        f"CSAT responses: {overview['csat_responses']:,} | "
        f"Avg CSAT: {overview['overall_avg_csat']:.2f}"
    )

    agent_df = compute_agent_metrics(filtered, closing_phrases=closing_phrases)

    min_contacts = thr.get("min_contacts")  # None → median

    # Auto-scale min_csat_responses when not set: max(3, 20% of avg CSAT responses per agent)
    min_csat_responses = thr.get("min_csat_responses")
    if min_csat_responses is None:
        n_with_csat = int((agent_df["csat_responses"] > 0).sum())
        avg_per_agent = overview["csat_responses"] / max(1, n_with_csat)
        min_csat_responses = max(3, round(avg_per_agent * 0.2))
        click.echo(f"   Auto min_csat_responses: {min_csat_responses} "
                   f"(20% of {avg_per_agent:.1f} avg CSAT responses/agent)")

    agent_df = flag_agents(
        agent_df,
        overall_avg_csat=overview["overall_avg_csat"],
        low_csat_offset=thr.get("low_csat_offset", 0.30),
        top_performer_offset=thr.get("top_performer_offset", 0.30),
        min_contacts=min_contacts,
        min_csat_responses=min_csat_responses,
    )
    agent_df = add_clusters(agent_df)

    n_pr = (agent_df["flag"] == "PRIORITY_REVIEW").sum()
    n_tp = (agent_df["flag"] == "TOP_PERFORMER").sum()
    click.echo(f"   Priority Review: {n_pr} agents | Top Performers: {n_tp} agents")

    drivers = compute_drivers(filtered)

    # ── Build report data model ───────────────────────────────────────────────
    click.echo(
        f"✍️  Generating coaching copy "
        f"({'LLM-assisted' if use_llm else 'rule-based'}) for {n_pr} agents…"
    )
    run_date = date.today()
    report_data = build_report(
        overview=overview,
        agent_df=agent_df,
        drivers=drivers,
        market=market,
        contact_reason=contact_reason_label,
        run_date=run_date,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        max_convo_examples=max_convo,
        convo_max_chars=convo_chars,
        contact_df=filtered,
    )

    # ── Render HTML ───────────────────────────────────────────────────────────
    click.echo("🖨  Rendering HTML…")
    template_path = ROOT / "report" / "template.html"
    html = render_html(report_data, template_path)

    # ── Save locally ──────────────────────────────────────────────────────────
    # Cloud Run jobs are ephemeral — write to /tmp when running in the cloud.
    # Locally, write to the configured output directory.
    if drive_folder_id:
        # Cloud mode: save to /tmp (ephemeral, then upload to Drive)
        local_path = Path("/tmp/report_output.html")
    else:
        # Local mode: save to outputs/ directory
        output_dir = ROOT / cfg["output"].get("dir", "outputs")
        filename = f"report_{market}_{run_date.strftime('%Y%m%d')}.html"
        local_path = output_dir / filename

    save_report(html, local_path)
    click.echo(f"\n✅ Report saved → {local_path}")

    # ── Upload to Google Drive ────────────────────────────────────────────────
    if drive_folder_id:
        from report.uploader import upload_to_drive, drive_filename
        fname = drive_filename(market, run_date, mode=drive_mode)
        click.echo(f"☁️  Uploading to Drive folder {drive_folder_id} as '{fname}'…")
        file_id, view_url = upload_to_drive(local_path, drive_folder_id, fname)
        click.echo(f"✅ Drive upload complete → {view_url}")

    if open_browser:
        import webbrowser
        webbrowser.open(local_path.as_uri())


if __name__ == "__main__":
    main()
