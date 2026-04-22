"""
Phase 1 — Data Loader

Fetch the Zomato dataset from Hugging Face with retry logic and caching.
Three loading strategies (tried in order):
  1. Primary:  Parquet files via pandas (smallest download, fastest)
  2. Fallback: HF `datasets` library with keep_in_memory mode
  3. Last resort: Direct CSV download via pandas

Functions:
    load_dataset() -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

from config import settings
from models import DataSourceError

logger = logging.getLogger(__name__)

# Maximum retry attempts per strategy
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds

# Ensure Hugging Face uses a writable cache directory
# (Render's free tier may not have a writable ~/.cache)
_HF_CACHE = str(settings.dataset_cache_dir / "huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)

# HuggingFace dataset URLs
_PARQUET_URLS = [
    "https://huggingface.co/api/datasets/ManikaSaini/zomato-restaurant-recommendation/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/ManikaSaini/zomato-restaurant-recommendation/parquet/default/train/1.parquet",
]

_DIRECT_CSV_URL = (
    "https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation"
    "/resolve/main/zomato.csv"
)


def load_dataset() -> pd.DataFrame:
    """
    Load the Zomato restaurant dataset from Hugging Face.

    Tries multiple strategies in order of efficiency:
      1. Parquet (fast, small downloads)
      2. HF datasets library (robust, but can have pickle issues)
      3. Direct CSV (large file, last resort)

    Returns:
        pd.DataFrame: Raw dataset as a pandas DataFrame.

    Raises:
        DataSourceError: If all strategies fail.
    """
    strategies = [
        ("Parquet", _try_parquet),
        ("HF datasets", _try_hf_datasets),
        ("Direct CSV", _try_direct_csv),
    ]

    for name, strategy in strategies:
        try:
            df = strategy()
            if df is not None and len(df) > 0:
                logger.info(f"✅ Dataset loaded via {name}: {len(df)} rows, {len(df.columns)} columns.")
                return df
        except Exception as exc:
            logger.warning(f"Strategy '{name}' failed: {type(exc).__name__}: {exc}")

    raise DataSourceError(
        "Failed to load dataset using all strategies (Parquet, HF datasets, CSV). "
        "Check network connectivity and Hugging Face availability."
    )


def _try_parquet() -> pd.DataFrame | None:
    """Download pre-built parquet files from HF (fastest, smallest)."""
    # Only load columns we actually need — saves ~70% memory
    _NEEDED_COLS = [
        "name", "city", "location", "cuisines", "rate",
        "votes", "average_cost_for_two", "approx_cost(for two people)",
        "online_order", "has_online_delivery", "aggregate_rating",
        "restaurant_name", "cuisine", "rating", "cost",
    ]

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Parquet] Downloading {len(_PARQUET_URLS)} parquet shards "
                f"(attempt {attempt}/{_MAX_RETRIES})..."
            )
            dfs = []
            for url in _PARQUET_URLS:
                pf = pd.read_parquet(url)
                # Select only columns that exist in this parquet file
                cols_to_keep = [c for c in pf.columns if c.lower() in [n.lower() for n in _NEEDED_COLS] or c.lower() in _NEEDED_COLS]
                if not cols_to_keep:
                    cols_to_keep = list(pf.columns)  # fallback: keep all
                dfs.append(pf[cols_to_keep] if cols_to_keep != list(pf.columns) else pf)

            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"[Parquet] Loaded: {len(df)} rows, {len(df.columns)} columns.")
            return df

        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                f"[Parquet] Attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait}s..."
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    return None


def _try_hf_datasets() -> pd.DataFrame | None:
    """Load using Hugging Face datasets library (in-memory to skip pickle)."""
    try:
        import datasets
    except ImportError:
        logger.warning("[HF datasets] Library not installed, skipping.")
        return None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"[HF datasets] Loading '{settings.dataset_id}' "
                f"(attempt {attempt}/{_MAX_RETRIES})..."
            )
            os.makedirs(_HF_CACHE, exist_ok=True)

            ds = datasets.load_dataset(
                settings.dataset_id,
                split="train",
                trust_remote_code=True,
                keep_in_memory=True,
                download_mode="force_redownload" if attempt > 1 else None,
            )
            return ds.to_pandas()

        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                f"[HF datasets] Attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait}s..."
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    return None


def _try_direct_csv() -> pd.DataFrame | None:
    """Download raw CSV directly (largest file, slowest, last resort)."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Direct CSV] Downloading ~725MB CSV "
                f"(attempt {attempt}/{_MAX_RETRIES})... this may take a while."
            )
            df = pd.read_csv(_DIRECT_CSV_URL)
            logger.info(f"[Direct CSV] Loaded: {len(df)} rows.")
            return df

        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                f"[Direct CSV] Attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait}s..."
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    return None
