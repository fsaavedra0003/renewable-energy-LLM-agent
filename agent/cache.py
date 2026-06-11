"""
agent/cache.py — Centralised cached resource loader

All module-level singletons live here so nothing is re-read from disk
on every agent call.  Import get_config() and get_pipeline_data() instead
of calling yaml.safe_load() or run_pipeline() directly inside node functions.

Usage
-----
from agent.cache import get_config, get_pipeline_data

cfg  = get_config()                          # read once, cached forever
docs, registry, telemetry_df, maintenance_df = get_pipeline_data()
"""

from __future__ import annotations

import functools
import logging
import yaml

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_config(path: str = "config.yaml") -> dict:
    """
    Load and cache config.yaml.

    lru_cache(maxsize=1) means the file is read exactly once per process
    lifetime regardless of how many nodes call get_config().  The path
    argument is part of the cache key so tests can pass an alternative path.
    """
    logger.debug(f"Loading config from {path}")
    with open(path) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def get_pipeline_data() -> tuple:
    """
    Run the ingestion pipeline exactly once and cache the result.

    Returns
    -------
    (documents, registry, telemetry_df, maintenance_df)
        The same 4-tuple that ingestion.pipeline.run_pipeline() returns.

    Note: lru_cache on a function with no arguments caches the single
    result permanently.  If you need to force a reload during testing,
    call get_pipeline_data.cache_clear() first.
    """
    from ingestion.pipeline import run_pipeline  # local import avoids circular deps

    logger.info("Running ingestion pipeline (first call — result will be cached)")
    return run_pipeline()
