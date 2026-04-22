"""
Phase 1 — Data Store

Persist and serve the cleaned DataFrame via pickle cache.

Functions:
    save(df, path)   -> None
    load(path)       -> pd.DataFrame
    cache_exists()   -> bool
    get_dataframe()  -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

_CACHE_FILE = "clean_restaurants.pkl"


def _cache_path() -> Path:
    """Resolve the full cache file path."""
    return settings.dataset_cache_dir / _CACHE_FILE


def cache_exists() -> bool:
    """Check whether a cached clean dataset file exists."""
    exists = _cache_path().is_file()
    if exists:
        size_mb = _cache_path().stat().st_size / (1024 * 1024)
        logger.debug(f"Cache found: {_cache_path()} ({size_mb:.1f} MB)")
    return exists


def save(df: pd.DataFrame, path: Path | None = None) -> None:
    """
    Save the cleaned DataFrame to a pickle file.

    Args:
        df: Cleaned DataFrame to persist.
        path: Optional override for the cache path.
    """
    target = path or _cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(str(target))
    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info(f"DataFrame cached: {target} ({size_mb:.1f} MB, {len(df)} rows)")


def load(path: Path | None = None) -> pd.DataFrame:
    """
    Load the cached DataFrame from pickle.

    Args:
        path: Optional override for the cache path.

    Returns:
        The cached clean DataFrame.

    Raises:
        FileNotFoundError: If the cache file doesn't exist.
    """
    target = path or _cache_path()

    if not target.is_file():
        raise FileNotFoundError(f"Cache file not found: {target}")

    df = pd.read_pickle(str(target))
    logger.info(f"Loaded cached DataFrame: {len(df)} rows from {target}")
    return df


def get_dataframe() -> pd.DataFrame:
    """
    Convenience function: load from cache if available,
    otherwise run the full ingestion pipeline.
    """
    if cache_exists():
        return load()

    # Lazy import to avoid circular dependency
    from data_ingestion.data_loader import load_dataset
    from data_ingestion.data_cleaner import clean

    raw_df = load_dataset()
    clean_df = clean(raw_df)
    save(clean_df)
    return clean_df


def clear_cache() -> None:
    """Delete the cached file (useful for testing or forced refresh)."""
    target = _cache_path()
    if target.is_file():
        target.unlink()
        logger.info(f"Cache cleared: {target}")
