"""
agent/tools/telemetry.py

Structured telemetry queries using pandas.
The DataFrame is loaded once and cached; all queries are exact computations.

Why pandas (not ChromaDB):
  - 8,844 rows of numeric data (energy_kwh, availability_pct, wind speed, irradiance)
  - Questions require aggregation, ranking, trend detection, fault code filtering
  - "Which assets underperformed last month?" needs argmin over a grouped sum
  - "What was WT-001 availability in March?" needs a date filter + mean
  - None of these are answerable by semantic similarity search

Fixes applied
-------------
- _asset_ids_from_results() and _asset_ids_from_text() removed — now imported
  from agent.tools.validators (were identical duplicates).
- get_underperforming_assets() no longer calls pd.read_csv() directly;
  it uses the cached registry from agent.cache.get_pipeline_data() instead.
- Module-level _df cache is still used for the telemetry DataFrame but is now
  populated from get_pipeline_data() to stay consistent with the single-load guarantee.
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

# Module-level cache — loaded once, reused across all agent calls
_df: pd.DataFrame | None = None


def _load(data_dir: str = "data") -> pd.DataFrame:
    global _df
    if _df is None:
        # Prefer the already-cached pipeline data to avoid a second disk read
        try:
            from agent.cache import get_pipeline_data
            _, _, _df, _ = get_pipeline_data()
        except Exception:
            # Fallback for standalone use (e.g. direct script invocation)
            from ingestion.pipeline import load_telemetry
            _df = load_telemetry(Path(data_dir) / "telemetry.csv")
    return _df


# ─────────────────────────────────────────────────────────────────────────────
# Primary query function
# ─────────────────────────────────────────────────────────────────────────────

def query_telemetry(
    question: str,
    tool_results: list[dict],
    data_dir: str = "data",
) -> dict:
    """
    Run a structured telemetry query.

    Resolves asset_id(s) from prior asset_lookup results first,
    then falls back to regex extraction from the question text.
    Infers date range from question keywords.
    Returns a structured error dict (not raises) if no rows match.
    """
    df = _load(data_dir)

    asset_ids  = asset_ids_from_results(tool_results) or asset_ids_from_text(question)
    date_start, date_end = infer_date_range(question)

    # ── build filter ─────────────────────────────────────────────────────────
    mask = pd.Series([True] * len(df), index=df.index)
    if asset_ids:
        mask &= df["asset_id"].isin(asset_ids)
    if date_start:
        mask &= df["date"] >= pd.Timestamp(date_start)
    if date_end:
        mask &= df["date"] <= pd.Timestamp(date_end)

    subset = df[mask].copy()

    # ── empty-result validation ───────────────────────────────────────────────
    empty_err = check_nonempty_result(subset, "telemetry")
    if empty_err:
        return {
            "status":  "no_data",
            "message": empty_err,
            "filters": {
                "asset_ids":   asset_ids,
                "date_start":  date_start,
                "date_end":    date_end,
            },
        }

    # ── aggregate stats ───────────────────────────────────────────────────────
    asset_label = ", ".join(asset_ids) if asset_ids else "all"
    result: dict = {
        "asset_id":            asset_label,
        "date_range":          f"{subset['date'].min().date()} to {subset['date'].max().date()}",
        "days":                len(subset),
        "avg_energy_kwh":      round(float(subset["energy_kwh"].mean()), 2),
        "total_energy_kwh":    round(float(subset["energy_kwh"].sum()), 2),
        "avg_availability_pct": round(float(subset["availability_pct"].mean()), 2),
        "min_availability_pct": round(float(subset["availability_pct"].min()), 2),
    }

    if subset["avg_wind_speed_ms"].notna().any():
        result["avg_wind_speed_ms"] = round(float(subset["avg_wind_speed_ms"].mean()), 2)
    if subset["irradiance_wm2"].notna().any():
        result["avg_irradiance_wm2"] = round(float(subset["irradiance_wm2"].mean()), 2)

    # ── fault summary ─────────────────────────────────────────────────────────
    faults = subset[subset["fault_code"].notna() & (subset["fault_code"] != "")]
    if not faults.empty:
        fault_summary = []
        for fc, grp in faults.groupby("fault_code"):
            fault_summary.append({
                "fault_code":  fc,
                "count":       len(grp),
                "description": grp["fault_description"].iloc[0],
                "first_date":  str(grp["date"].min().date()),
                "last_date":   str(grp["date"].max().date()),
            })
        result["fault_events"] = fault_summary
        result["fault_count"]  = len(faults)
    else:
        result["fault_events"] = []
        result["fault_count"]  = 0

    # ── monthly trend ─────────────────────────────────────────────────────────
    subset = subset.copy()
    subset["month"] = subset["date"].dt.to_period("M")
    monthly = (
        subset.groupby("month")
        .agg(avg_energy=("energy_kwh", "mean"), avg_avail=("availability_pct", "mean"))
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    result["monthly_trend"] = monthly.to_dict("records")

    return {"status": "ok", "data": result}


def get_underperforming_assets(
    location: str | None = None,
    data_dir: str = "data",
    comparison_month: str | None = None,
) -> dict:
    """
    Detect assets with abnormal energy output degradation since March 2024.

    Methodology (seasonality-normalised, within asset-class)
    ----------------------------------------------------------
    Wind output naturally declines and solar output naturally rises from
    winter (Jan-Feb) into summer (Mar-Jun) — a raw "% change" comparison
    therefore flags the entire wind fleet (or none of the solar fleet).

    Instead, for each asset we compute its own seasonal ratio:

        ratio = mean(Mar-Jun energy_kwh) / mean(Jan-Feb energy_kwh)

    This ratio captures each asset's *own* seasonal trend. We then compute
    a z-score of this ratio *within its asset class* (wind turbines vs solar
    plants compared separately, since they have opposite seasonal directions
    and different baselines). Assets whose ratio is an outlier on the LOW
    side relative to their peers (z <= z_threshold) have a Mar-Jun trend
    that diverges from what the rest of the fleet experienced — i.e. a
    genuine anomaly on top of (not explained by) normal seasonality.

    comparison_month, if provided, restricts the reporting window for
    displayed averages to a single month (e.g. for "last month" queries),
    while anomaly detection always runs on the full Jan-Feb vs Mar-Jun
    comparison (the structural signal).

    Fix: asset metadata now comes from the cached registry (get_pipeline_data)
    instead of a direct pd.read_csv() call, keeping a single source of truth.
    """
    df = _load(data_dir).copy()

    # ── asset metadata from cache (not re-read from disk) ─────────────────────
    try:
        from agent.cache import get_pipeline_data
        _, registry, _, _ = get_pipeline_data()
        asset_meta = pd.DataFrame(registry.values())[["asset_id", "location", "type"]]
    except Exception:
        # Fallback for standalone use
        asset_meta = pd.read_csv(Path(data_dir) / "assets.csv")[["asset_id", "location", "type"]]
        asset_meta["asset_id"] = asset_meta["asset_id"].str.strip().str.upper()

    df = df.merge(asset_meta, on="asset_id", how="left")

    if location:
        loc_norm = location.lower().strip()
        exact    = df["location"].str.lower().str.strip() == loc_norm
        if exact.any():
            df = df[exact]
        else:
            df = df[df["location"].str.lower().str.contains(loc_norm, na=False)]
        if df.empty:
            return {
                "status":  "no_data",
                "message": (
                    f"No assets found for location '{location}'. "
                    f"Available locations: Northern Spain, Castile, Galicia, Aragon, "
                    f"Navarre, La Rioja, Southern Spain, Andalusia, Murcia, Extremadura."
                ),
            }
        logger.info(f"Location filter '{location}': {df['asset_id'].nunique()} assets")

    df["month_num"] = df["date"].dt.month

    # ── per-asset seasonal ratio: Mar-Jun avg / Jan-Feb avg ───────────────────
    pre  = df[df["month_num"].isin([1, 2])].groupby("asset_id")["energy_kwh"].mean()
    post = df[df["month_num"].isin([3, 4, 5, 6])].groupby("asset_id")["energy_kwh"].mean()

    seasonal = pd.concat([pre.rename("pre_avg"), post.rename("post_avg")], axis=1).dropna()
    seasonal["ratio"] = seasonal["post_avg"] / seasonal["pre_avg"]

    # asset_class for grouping (wind vs solar — opposite seasonal directions)
    asset_class = df.drop_duplicates("asset_id").set_index("asset_id")["type"]
    seasonal["asset_class"] = asset_class

    z_threshold = -1.5  # isolates assets clearly separated from their class peers
    flagged_ids: list[str] = []
    z_scores: dict[str, float] = {}

    for cls, grp in seasonal.groupby("asset_class"):
        if len(grp) < 3 or grp["ratio"].std() == 0 or pd.isna(grp["ratio"].std()):
            continue
        z = (grp["ratio"] - grp["ratio"].mean()) / grp["ratio"].std()
        for aid, zval in z.items():
            z_scores[aid] = round(float(zval), 2)
            if zval <= z_threshold:
                flagged_ids.append(aid)

    # sort flagged assets by severity (most negative z first)
    flagged_ids.sort(key=lambda a: z_scores[a])

    # ── reporting window for displayed averages ───────────────────────────────
    df["month"] = df["date"].dt.to_period("M").astype(str)
    if comparison_month:
        late_mask  = df["month"] == comparison_month
        late_label = comparison_month
    else:
        late_mask  = df["month_num"].isin([3, 4, 5, 6])
        late_label = "2024-03 to 2024-06"

    late_avg    = df[late_mask].groupby("asset_id")["energy_kwh"].mean()
    late_faults = df[late_mask]

    results: list[dict] = []
    for asset_id in flagged_ids:
        row = seasonal.loc[asset_id]
        asset_faults = late_faults[
            (late_faults["asset_id"] == asset_id) &
            late_faults["fault_code"].notna() &
            (late_faults["fault_code"] != "")
        ]
        fault_summary = (
            asset_faults.groupby("fault_code")["fault_description"]
            .first().reset_index()
            .rename(columns={"fault_description": "description"})
            .to_dict("records")
        )
        reporting_avg = late_avg.get(asset_id)
        results.append({
            "asset_id":            asset_id,
            "asset_class":         row["asset_class"],
            "z_score":             z_scores[asset_id],
            "jan_feb_avg_kwh":     round(float(row["pre_avg"]), 0),
            "post_march_avg_kwh":  round(float(row["post_avg"]), 0),
            "seasonal_ratio":      round(float(row["ratio"]), 3),
            "reporting_window":    late_label,
            "reporting_avg_kwh":   round(float(reporting_avg), 0) if pd.notna(reporting_avg) else None,
            "recent_fault_codes":  fault_summary,
        })

    return {
        "status":                "ok",
        "location_filter":       location,
        "method": (
            "Seasonality-normalised z-score: each asset's (Mar-Jun avg / Jan-Feb avg) "
            "ratio is compared against its own asset-class peers (wind vs solar "
            f"compared separately). Assets with z <= {z_threshold} have a Mar-Jun "
            "trend that diverges from the rest of their class beyond what normal "
            "seasonal variation explains."
        ),
        "z_threshold":           z_threshold,
        "underperforming_count": len(results),
        "underperforming_assets": results,
    }


def get_low_availability_assets(
    date_start: str = "2024-03-01",
    date_end:   str = "2024-06-30",
    threshold:  float = 90.0,
    data_dir:   str = "data",
) -> dict:
    """Return assets with mean availability_pct below threshold in the given window."""
    df   = _load(data_dir)
    mask = (df["date"] >= pd.Timestamp(date_start)) & (df["date"] <= pd.Timestamp(date_end))
    subset = df[mask]

    empty_err = check_nonempty_result(subset, "telemetry (availability)")
    if empty_err:
        return {"status": "no_data", "message": empty_err}

    avail = subset.groupby("asset_id")["availability_pct"].mean().round(2)
    low   = avail[avail < threshold].sort_values()

    results: list[dict] = []
    for asset_id, mean_avail in low.items():
        asset_rows  = subset[subset["asset_id"] == asset_id]
        min_avail   = round(float(asset_rows["availability_pct"].min()), 2)
        fault_count = asset_rows["fault_code"].notna().sum()
        results.append({
            "asset_id":               asset_id,
            "mean_availability_pct":  mean_avail,
            "min_availability_pct":   min_avail,
            "fault_events_in_period": int(fault_count),
        })

    return {
        "status":               "ok",
        "date_range":           f"{date_start} to {date_end}",
        "threshold_pct":        threshold,
        "low_availability_count": len(results),
        "assets":               results,
    }


def get_fault_history(asset_id: str, data_dir: str = "data") -> dict:
    """Return all fault events for a specific asset, sorted by date."""
    df     = _load(data_dir)
    faults = df[
        (df["asset_id"] == asset_id) &
        df["fault_code"].notna() &
        (df["fault_code"] != "")
    ].sort_values("date")

    if faults.empty:
        return {"status": "ok", "asset_id": asset_id, "fault_events": [], "fault_count": 0}

    events = faults[["date", "fault_code", "fault_description", "availability_pct"]].copy()
    events["date"] = events["date"].dt.date.astype(str)
    return {
        "status":       "ok",
        "asset_id":     asset_id,
        "fault_count":  len(events),
        "fault_events": events.to_dict("records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Date range inference moved to agent.tools.validators.infer_date_range
# (was duplicated here and in maintenance.py with slightly different
# keyword sets — now a single shared superset implementation)
# ─────────────────────────────────────────────────────────────────────────────
