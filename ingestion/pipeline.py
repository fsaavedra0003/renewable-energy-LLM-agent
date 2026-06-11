"""
ingestion/pipeline.py  —  Task 1.1: Data Ingestion & Preprocessing Pipeline

Strategy per file
─────────────────
manual_excerpts.txt  → Document objects  → ChromaDB (only embeddable source)
assets.csv           → Python dict        → in-memory registry for fast lookup + fuzzy match
telemetry.csv        → pandas DataFrame   → structured queries via agent/tools/telemetry.py
maintenance_logs.csv → pandas DataFrame   → structured queries via agent/tools/maintenance.py

Rationale
─────────
Embedding is only useful when the query requires semantic similarity over unstructured prose.
  - The manual has 2,100 words of genuine OEM documentation across 4 sections — embed it.
  - Assets has 50 rows of key-value fields — a plain dict is faster and exact.
  - Telemetry has 8,844 numeric rows — aggregation and ranking need pandas, not vectors.
  - Maintenance has 188 rows with only 15 unique template descriptions — pandas str.contains()
    on fault codes is exact and instant; embedding near-identical strings adds zero value.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_DATE = date(2024, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Document dataclass (used only by manual loader)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        fc = self.metadata.get("fault_code", "overview")
        return f"Document(fault_code={fc}, chars={len(self.page_content)})"


# ─────────────────────────────────────────────────────────────────────────────
# 1. assets.csv  →  validated dict registry
# ─────────────────────────────────────────────────────────────────────────────

def load_assets(path: Path) -> dict[str, dict]:
    """
    Load and validate assets.csv.
    Returns dict[asset_id -> row_dict] for O(1) lookup.
    Adds derived fields: age_years, asset_class.
    """
    df = pd.read_csv(path)

    required = ["asset_id", "type", "location", "capacity_mw",
                "manufacturer", "model", "install_date", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"assets.csv missing columns: {missing}")

    df["asset_id"] = df["asset_id"].str.strip().str.upper()
    df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce").dt.date
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

    dupes = df[df["asset_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Duplicate asset_ids: {dupes['asset_id'].tolist()}")

    # Warn on unparseable install_dates (mirrors telemetry behaviour)
    null_installs = df["install_date"].isnull().sum()
    if null_installs:
        logger.warning(f"Assets: {null_installs} rows with unparseable install_date")

    df["age_years"] = df["install_date"].apply(
        lambda d: round((REFERENCE_DATE - d).days / 365.25, 2) if pd.notnull(d) else None
    )
    df["asset_class"] = df["type"].map({"wind_turbine": "wind", "solar_plant": "solar"})

    registry = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        d["install_date"] = str(d["install_date"])
        registry[d["asset_id"]] = d

    logger.info(
        f"Assets: {len(registry)} loaded | "
        f"wind={sum(1 for v in registry.values() if v['asset_class']=='wind')} "
        f"solar={sum(1 for v in registry.values() if v['asset_class']=='solar')}"
    )
    return registry


def resolve_asset_id(query: str, registry: dict[str, dict]) -> dict | None:
    """
    Resolve a potentially misspelled asset_id.

    Strategy:
    1. Exact match (O(1)).
    2. Levenshtein distance ≤ 1 within the same asset class prefix (WT- or PV-).
       Tighter threshold than original (was ≤ 2 across all IDs) to prevent
       WT-001 matching WT-003 — a distance of 2 is 33% of a 6-char ID.

    Returns asset dict or None.
    """
    query = query.strip().upper()
    if query in registry:
        return registry[query]

    try:
        from rapidfuzz.distance import Levenshtein

        # Only fuzzy-match within the same asset class to avoid cross-class false positives
        prefix_match = re.match(r"^(WT|PV)-", query)
        if prefix_match:
            prefix = prefix_match.group(0)
            class_ids = [aid for aid in registry if aid.startswith(prefix)]
        else:
            class_ids = list(registry.keys())

        if class_ids:
            candidates = [(aid, Levenshtein.distance(query, aid)) for aid in class_ids]
            best_id, best_dist = min(candidates, key=lambda x: x[1])
            if best_dist <= 1:
                logger.info(f"Fuzzy matched '{query}' → '{best_id}' (dist={best_dist})")
                return registry[best_id]

    except ImportError:
        # rapidfuzz not installed — fall back to prefix scan
        prefix = [aid for aid in registry if aid.startswith(query[:4])]
        if len(prefix) == 1:
            return registry[prefix[0]]

    logger.warning(f"Could not resolve asset_id: '{query}'")
    return None


def asset_context_string(asset: dict) -> str:
    """One-line context prefix for LLM prompts."""
    return (
        f"[{asset['asset_id']} | {asset['name']} | "
        f"{asset['type'].replace('_', ' ')} | {asset['location']} | "
        f"{asset['manufacturer']} {asset['model']} | "
        f"{asset['capacity_mw']} MW | installed {asset['install_date']} "
        f"({asset.get('age_years', '?')} yrs) | {asset['status']}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. telemetry.csv  →  validated pandas DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_telemetry(path: Path) -> pd.DataFrame:
    """
    Load and validate telemetry.csv.
    Returns a clean DataFrame. All querying is done by agent/tools/telemetry.py.

    NOT embedded — 8,844 numeric rows.
    Every question about telemetry is a structured computation:
    aggregation, ranking, trend detection, fault filtering.
    pandas handles these exactly and instantly.
    """
    df = pd.read_csv(path)
    df["asset_id"] = df["asset_id"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df["availability_pct"] = pd.to_numeric(df["availability_pct"], errors="coerce")
    df["avg_wind_speed_ms"] = pd.to_numeric(df["avg_wind_speed_ms"], errors="coerce")
    df["irradiance_wm2"] = pd.to_numeric(df["irradiance_wm2"], errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")

    null_dates = df["date"].isnull().sum()
    if null_dates:
        logger.warning(f"Telemetry: {null_dates} rows with unparseable dates dropped")
        df = df[df["date"].notnull()]

    logger.info(
        f"Telemetry: {len(df)} rows | "
        f"{df['asset_id'].nunique()} assets | "
        f"{df['date'].min().date()} → {df['date'].max().date()} | "
        f"fault rows: {df['fault_code'].notna().sum()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. maintenance_logs.csv  →  validated pandas DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_maintenance(path: Path) -> pd.DataFrame:
    """
    Load and validate maintenance_logs.csv.
    Returns a clean DataFrame. All querying is done by agent/tools/maintenance.py.

    NOT embedded — only 15 unique description templates across 188 rows.
    All realistic questions are structural queries:
      - history for a specific asset  →  filter by asset_id + sort by date
      - any gearbox work?             →  str.contains on description/parts_replaced
      - total cost?                   →  sum(cost_eur)
      - is maintenance due?           →  max(date) + 180 day cadence
    pandas str.contains() on fault codes and parts is exact; embedding
    near-identical boilerplate strings adds zero retrieval value.
    """
    df = pd.read_csv(path)
    df["asset_id"] = df["asset_id"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cost_eur"] = pd.to_numeric(df["cost_eur"], errors="coerce")
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["description"] = df["description"].fillna("").astype(str)
    df["parts_replaced"] = df["parts_replaced"].fillna("none").astype(str)

    logger.info(
        f"Maintenance: {len(df)} rows | "
        f"{df['asset_id'].nunique()} assets | "
        f"scheduled={df['type'].eq('scheduled').sum()} "
        f"corrective={df['type'].eq('corrective').sum()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. manual_excerpts.txt  →  Document objects for ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def load_manual(path: Path, chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    """
    Parse manual_excerpts.txt into semantic chunks for ChromaDB embedding.

    THE ONLY SOURCE THAT GETS EMBEDDED.
    2,100 words of genuine OEM prose across 4 sections and 12 fault codes.
    Semantic search is the correct retrieval method because:
      - Questions are natural language ("what does E-1001 mean?",
        "what causes gearbox overheating?", "escalation procedure for HIGH faults")
      - Answers require understanding meaning across paragraphs, not exact field lookup
      - No amount of pandas filtering can answer "what are the corrective actions for..."

    Chunking strategy:
      - FAULT CODE blocks → one Document each (natural semantic boundary, ~300-500 chars)
      - PERFORMANCE OVERVIEW sections → fixed-size sub-chunks with overlap
      - GENERAL OPERATIONS sections → fixed-size sub-chunks with overlap
    """
    text = path.read_text(encoding="utf-8")
    docs: list[Document] = []

    # Section metadata map — used to tag each chunk with manufacturer/model
    section_markers = [
        ("SECTION 1", "Vestas", "V150-4.5"),
        ("SECTION 2", "Siemens Gamesa", "SG-6.6-170"),
        ("SECTION 3", "LONGi Solar", "Hi-MO 6"),
        ("SECTION 4", "General", "All models"),
    ]

    def _section_meta(char_pos: int) -> tuple[str, str]:
        mfr, model = "Unknown", "Unknown"
        for marker, m, mo in section_markers:
            if text.find(marker) != -1 and text.find(marker) <= char_pos:
                mfr, model = m, mo
        return mfr, model

    # ── Fault code blocks (primary retrieval target) ─────────────────────────
    fault_pattern = re.compile(
        r"(FAULT CODE: E-\d+.*?)(?=FAULT CODE: E-\d+|^-{10,}|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in fault_pattern.finditer(text):
        chunk = m.group(0).strip()
        if len(chunk) < 50:
            continue
        fc_match = re.search(r"FAULT CODE: (E-\d+)", chunk)
        fault_code = fc_match.group(1) if fc_match else "unknown"
        mfr, model = _section_meta(m.start())

        docs.append(Document(
            page_content=f"[OEM Manual | {mfr} {model}]\n{chunk}",
            metadata={
                "doc_type": "manual",
                "chunk_type": "fault_code",
                "fault_code": fault_code,
                "manufacturer": mfr,
                "model": model,
            },
        ))

    # ── Performance overview + general ops sections ──────────────────────────
    narrative_pattern = re.compile(
        r"(\d+\.\d+\s+(?:PERFORMANCE OVERVIEW|MAINTENANCE SCHEDULING|"
        r"DATA QUALITY|ESCALATION MATRIX).*?)(?=\d+\.\d+\s+[A-Z]|-{10,}|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in narrative_pattern.finditer(text):
        chunk = m.group(0).strip()
        if len(chunk) < 100:
            continue
        mfr, model = _section_meta(m.start())
        section_title = chunk.split("\n")[0].strip()

        for i in range(0, len(chunk), chunk_size - chunk_overlap):
            sub = chunk[i: i + chunk_size].strip()
            if len(sub) < 80:
                continue
            docs.append(Document(
                page_content=f"[OEM Manual | {mfr} {model}]\n{sub}",
                metadata={
                    "doc_type": "manual",
                    "chunk_type": "narrative",
                    "fault_code": None,
                    "manufacturer": mfr,
                    "model": model,
                    "section": section_title,
                },
            ))

    logger.info(
        f"Manual: {len(docs)} chunks | "
        f"fault_code={sum(1 for d in docs if d.metadata['chunk_type']=='fault_code')} | "
        f"narrative={sum(1 for d in docs if d.metadata['chunk_type']=='narrative')}"
    )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str | Path = "data",
    config_path: str | Path = "config.yaml",
) -> tuple[list[Document], dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """
    Run the full ingestion pipeline.

    Returns
    -------
    manual_docs  : Documents ready for ChromaDB embedding (manual only)
    registry     : dict[asset_id -> asset row] for agent lookups
    telemetry_df : clean DataFrame for pandas queries
    maintenance_df: clean DataFrame for pandas queries
    """
    import yaml
    data_dir = Path(data_dir)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    chunking = cfg["chunking"]

    logger.info("=== Ingestion pipeline starting ===")

    registry = load_assets(data_dir / "assets.csv")
    telemetry_df = load_telemetry(data_dir / "telemetry.csv")
    maintenance_df = load_maintenance(data_dir / "maintenance_logs.csv")
    manual_docs = load_manual(
        data_dir / "manual_excerpts.txt",
        chunk_size=chunking["manual_chunk_size"],
        chunk_overlap=chunking["manual_chunk_overlap"],
    )

    logger.info(
        f"=== Pipeline complete | "
        f"manual_docs={len(manual_docs)} | "
        f"assets={len(registry)} | "
        f"telemetry_rows={len(telemetry_df)} | "
        f"maintenance_rows={len(maintenance_df)} ==="
    )
    return manual_docs, registry, telemetry_df, maintenance_df


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up helper
# ─────────────────────────────────────────────────────────────────────────────

def warm_up(data_dir: str | Path = "data", config_path: str | Path = "config.yaml") -> None:
    """
    Pre-load the pipeline result into agent.cache so the first agent call
    does not pay the full pipeline startup cost.
    """
    from agent.cache import get_pipeline_data
    get_pipeline_data()
    logger.info("Pipeline warmed up and cached.")
