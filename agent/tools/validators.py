"""
agent/tools/validators.py — Shared per-tool input validation helpers

Each tool node calls these helpers at the TOP of its function, before
touching any data source.  If a check fails the helper returns a plain
error string; the tool node wraps that in a structured error dict and
returns early without executing the query.

This module also holds the shared asset-ID extraction helpers that were
previously duplicated verbatim in telemetry.py and maintenance.py.

Pattern
-------
    from agent.tools.validators import (
        check_upstream_asset, check_nonempty_result,
        asset_ids_from_results, asset_ids_from_text,
    )

    def telemetry_node(state):
        err = check_upstream_asset(state.get("tool_results", []))
        if err:
            return _err("telemetry_tool", err, state)
        ...
        df = query(...)
        err = check_nonempty_result(df, "telemetry")
        if err:
            return _err("telemetry_tool", err, state)
"""

from __future__ import annotations

import re
from datetime import date


# ---------------------------------------------------------------------------
# Upstream asset check
# ---------------------------------------------------------------------------

def check_upstream_asset(tool_results: list[dict]) -> str | None:
    """
    Return an error string if a previous asset_lookup failed, else None.

    Telemetry and maintenance tools should call this first.  If asset_lookup
    returned an error there is no valid asset_id to filter on — running the
    pandas query would return the entire dataset or raise a KeyError, both
    of which are worse than an explicit early exit.
    """
    for result in tool_results:
        if result.get("tool") == "asset_lookup" and result.get("error"):
            return (
                f"Skipped — asset lookup reported: {result['error']}. "
                "Cannot query without a valid asset ID."
            )
    return None


# ---------------------------------------------------------------------------
# Asset ID extraction (shared — previously duplicated in telemetry.py and maintenance.py)
# ---------------------------------------------------------------------------

def asset_ids_from_results(tool_results: list[dict]) -> list[str]:
    """
    Extract asset IDs from prior asset_lookup tool results.

    Handles both the single-asset case (data.asset_id) and the
    multi-asset case (data.assets[]).
    """
    for r in tool_results:
        if r.get("tool") == "asset_lookup" and r.get("data"):
            data = r["data"]
            if "assets" in data:
                return [a["asset_id"] for a in data["assets"] if "asset_id" in a]
            if "asset_id" in data:
                return [data["asset_id"]]
    return []


def asset_ids_from_text(text: str) -> list[str]:
    """Extract all asset IDs from free text (fallback when no lookup result)."""
    matches = re.findall(r"\b(?:WT|PV)-\d+\b", text, re.IGNORECASE)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        upper = m.upper()
        if upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result


# ---------------------------------------------------------------------------
# extract_all_asset_ids (used by agent.py asset_lookup_node)
# ---------------------------------------------------------------------------

def extract_all_asset_ids(question: str) -> list[str]:
    """
    Return ALL asset IDs found in the question string (upper-cased).

    A question like "Compare WT-001 and WT-002" returns ["WT-001", "WT-002"].
    """
    return asset_ids_from_text(question)


# ---------------------------------------------------------------------------
# Date range inference (shared — previously duplicated in telemetry.py
# and maintenance.py with slightly different keyword sets; this is the
# superset of both)
# ---------------------------------------------------------------------------

def infer_date_range(question: str) -> tuple[str, str]:
    """
    Infer a (date_start, date_end) ISO range from question keywords.
    Dataset covers January-June 2024.
    """
    q = question.lower()
    months = {
        "january":  ("2024-01-01", "2024-01-31"),
        "february": ("2024-02-01", "2024-02-29"),
        "march":    ("2024-03-01", "2024-03-31"),
        "april":    ("2024-04-01", "2024-04-30"),
        "may":      ("2024-05-01", "2024-05-31"),
        "june":     ("2024-06-01", "2024-06-30"),
    }
    if "last month" in q:
        return "2024-06-01", "2024-06-30"
    if "past 6 months" in q or "last 6 months" in q or "6 months" in q or "past 6" in q:
        return "2024-01-01", "2024-06-30"
    if "past 3 months" in q or "last 3 months" in q or "past 3" in q or "3 months" in q or "q2" in q:
        return "2024-04-01", "2024-06-30"
    if "q1" in q:
        return "2024-01-01", "2024-03-31"
    for name, (start, end) in months.items():
        if name in q:
            return start, end
    return "2024-01-01", "2024-06-30"


# ---------------------------------------------------------------------------
# Date range check
# ---------------------------------------------------------------------------

def check_date_range(date_start: str, date_end: str) -> str | None:
    """
    Return an error string if the date range is empty or malformed, else None.
    """
    try:
        start = date.fromisoformat(date_start)
        end   = date.fromisoformat(date_end)
    except ValueError as exc:
        return f"Invalid date format: {exc}"

    if start > end:
        return (
            f"Date range is empty: start {date_start} is after end {date_end}. "
            "The dataset covers January–June 2024."
        )
    return None


# ---------------------------------------------------------------------------
# Empty result check
# ---------------------------------------------------------------------------

def check_nonempty_result(df, label: str) -> str | None:
    """
    Return an error string if a pandas DataFrame has zero rows, else None.
    """
    try:
        if df is None or len(df) == 0:
            return (
                f"No {label} records found for the specified filters. "
                "Try broadening the date range or checking the asset ID."
            )
    except Exception:
        return f"Could not determine {label} result size — query may have failed."
    return None


# ---------------------------------------------------------------------------
# RAG minimum score check
# ---------------------------------------------------------------------------

def check_rag_results(results: list[dict], min_score: float = 0.30) -> str | None:
    """
    Return an error string if ChromaDB returned no results above min_score.
    """
    if not results:
        return "No documents retrieved from the manual for this query."

    best = max((r.get("score", 0.0) for r in results), default=0.0)
    if best < min_score:
        return (
            f"Retrieved documents have low relevance (best score: {best:.2f}, "
            f"threshold: {min_score}). The OEM manual may not cover this topic."
        )
    return None
