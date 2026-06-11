"""
agent/tools/maintenance.py

Structured maintenance log queries using pandas.
The DataFrame is loaded once and cached.

Why pandas (not ChromaDB):
  - 188 rows, only 15 unique description templates
  - All realistic questions are structural:
      "history of WT-011"     → filter asset_id + sort date
      "any gearbox work?"     → str.contains on description / parts_replaced
      "total cost for PV-019?"→ sum(cost_eur) where asset_id='PV-019'
      "is maintenance due?"   → max(date) + 180-day cadence check
  - pandas str.contains() on fault codes is exact and instant
  - Embedding 15 near-identical boilerplate templates adds zero retrieval value

Fixes applied
-------------
- _asset_ids_from_results() and _asset_ids_from_text() removed — now imported
  from agent.tools.validators (were identical duplicates).
- _load() now prefers get_pipeline_data() cache over a fresh disk read.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from agent.tools.validators import (
    check_nonempty_result,
    asset_ids_from_results,
    asset_ids_from_text,
    infer_date_range,
)

logger = logging.getLogger(__name__)

_df: pd.DataFrame | None = None


def _load(data_dir: str = "data") -> pd.DataFrame:
    global _df
    if _df is None:
        try:
            from agent.cache import get_pipeline_data
            _, _, _, _df = get_pipeline_data()
        except Exception:
            from ingestion.pipeline import load_maintenance
            _df = load_maintenance(Path(data_dir) / "maintenance_logs.csv")
    return _df


# ─────────────────────────────────────────────────────────────────────────────
# Primary query function
# ─────────────────────────────────────────────────────────────────────────────

def query_maintenance(
    question: str,
    tool_results: list[dict],
    data_dir: str = "data",
) -> dict:
    """
    Query maintenance logs for one or more specific assets.

    Resolves asset_id(s) from prior asset_lookup results first,
    then falls back to regex extraction from the question.
    Returns a structured error dict (not raises) if no rows match.
    """
    df = _load(data_dir)

    asset_ids = asset_ids_from_results(tool_results) or asset_ids_from_text(question)

    if not asset_ids:
        return {
            "status":  "error",
            "message": (
                "No asset_id found. Please specify an asset like WT-001 or PV-005. "
                "The asset_lookup tool should run before this tool."
            ),
        }

    date_start, date_end = infer_date_range(question)

    mask = df["asset_id"].isin(asset_ids)
    if date_start:
        mask &= df["date"] >= pd.Timestamp(date_start)
    if date_end:
        mask &= df["date"] <= pd.Timestamp(date_end)

    subset = df[mask].copy().sort_values("date")

    # ── empty-result validation ───────────────────────────────────────────────
    asset_label = ", ".join(asset_ids)
    empty_err   = check_nonempty_result(subset, f"maintenance records for '{asset_label}'")
    if empty_err:
        return {
            "status":  "no_data",
            "message": empty_err,
            "filters": {
                "asset_ids":  asset_ids,
                "date_start": date_start,
                "date_end":   date_end,
            },
        }

    # ── build event list ──────────────────────────────────────────────────────
    events: list[dict] = []
    for _, row in subset.iterrows():
        events.append({
            "log_id":         row["log_id"],
            "asset_id":       row["asset_id"],
            "date":           str(row["date"].date()),
            "type":           row["type"],
            "description":    str(row["description"])[:300],
            "technician":     row.get("technician", "unknown"),
            "duration_hours": row.get("duration_hours"),
            "parts_replaced": str(row["parts_replaced"]),
            "cost_eur":       round(float(row["cost_eur"]), 2) if pd.notna(row.get("cost_eur")) else None,
        })

    # ── summary stats ─────────────────────────────────────────────────────────
    scheduled  = subset[subset["type"] == "scheduled"]
    corrective = subset[subset["type"] == "corrective"]
    last_event = subset.iloc[-1]

    # next scheduled: use FULL history for these assets (not just the query window)
    all_asset_records = df[df["asset_id"].isin(asset_ids)]
    all_scheduled     = all_asset_records[all_asset_records["type"] == "scheduled"]
    last_sched_date   = all_scheduled["date"].max() if not all_scheduled.empty else None
    next_scheduled    = None
    if last_sched_date is not None and pd.notna(last_sched_date):
        next_scheduled = str((last_sched_date + pd.Timedelta(days=180)).date())

    overdue = False
    if last_sched_date is not None and pd.notna(last_sched_date):
        days_since = (pd.Timestamp("2024-07-01") - last_sched_date).days
        overdue    = days_since > 180

    fault_codes_mentioned = re.findall(r"E-\d+", " ".join(subset["description"].tolist()))
    parts_used            = subset["parts_replaced"].value_counts().to_dict()

    summary = {
        "asset_id":                  asset_label,
        "date_range":                f"{subset['date'].min().date()} to {subset['date'].max().date()}",
        "total_events":              len(subset),
        "scheduled_count":           len(scheduled),
        "corrective_count":          len(corrective),
        "total_cost_eur":            round(float(subset["cost_eur"].sum()), 2),
        "avg_cost_eur":              round(float(subset["cost_eur"].mean()), 2),
        "total_duration_hours":      round(float(subset["duration_hours"].sum()), 2),
        "last_maintenance_date":     str(last_event["date"].date()),
        "last_maintenance_type":     last_event["type"],
        "estimated_next_scheduled":  next_scheduled,
        "maintenance_overdue":       overdue,
        "fault_codes_in_logs":       list(set(fault_codes_mentioned)),
        "parts_used":                parts_used,
        "events":                    events,
    }

    return {"status": "ok", "data": summary}


def search_by_fault_code(fault_code: str, data_dir: str = "data") -> dict:
    """Find all maintenance events triggered by a specific fault code."""
    df     = _load(data_dir)
    mask   = df["description"].str.contains(fault_code, case=False, na=False)
    subset = df[mask].sort_values("date")

    if subset.empty:
        return {
            "status":  "no_data",
            "message": f"No maintenance events found for fault code '{fault_code}'.",
        }

    events: list[dict] = []
    for _, row in subset.iterrows():
        events.append({
            "log_id":         row["log_id"],
            "asset_id":       row["asset_id"],
            "date":           str(row["date"].date()),
            "description":    str(row["description"])[:200],
            "parts_replaced": str(row["parts_replaced"]),
            "cost_eur":       round(float(row["cost_eur"]), 2) if pd.notna(row.get("cost_eur")) else None,
        })

    return {
        "status":         "ok",
        "fault_code":     fault_code,
        "event_count":    len(events),
        "total_cost_eur": round(float(subset["cost_eur"].sum()), 2),
        "events":         events,
    }


def search_by_part(part_keyword: str, data_dir: str = "data") -> dict:
    """Find all maintenance events where a specific part was replaced."""
    df     = _load(data_dir)
    mask   = df["parts_replaced"].str.contains(part_keyword, case=False, na=False)
    subset = df[mask].sort_values("date")

    if subset.empty:
        return {
            "status":  "no_data",
            "message": f"No maintenance events found involving part '{part_keyword}'.",
        }

    return {
        "status":          "ok",
        "part_keyword":    part_keyword,
        "event_count":     len(subset),
        "assets_affected": subset["asset_id"].unique().tolist(),
        "total_cost_eur":  round(float(subset["cost_eur"].sum()), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Date range inference moved to agent.tools.validators.infer_date_range
# (was duplicated here and in telemetry.py — now a single shared implementation)
# ─────────────────────────────────────────────────────────────────────────────
