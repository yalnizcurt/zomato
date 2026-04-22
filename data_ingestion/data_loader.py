"""
Phase 1 — Data Loader

Fetch the Zomato dataset from Hugging Face with retry logic and caching.

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

# Maximum retry attempts for dataset download
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds

# Ensure Hugging Face uses a writable cache directory
# (Render's free tier may not have a writable ~/.cache)
_HF_CACHE = str(settings.dataset_cache_dir / "huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)


def load_dataset() -> pd.DataFrame:
    """
    Load the Zomato restaurant dataset from Hugging Face.

    Implements exponential backoff on failure (up to 3 attempts).

    Returns:
        pd.DataFrame: Raw dataset as a pandas DataFrame.

    Raises:
        DataSourceError: If the dataset cannot be loaded after all retries.
    """
    import datasets  # Lazy import — only needed when cache miss

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"Loading dataset '{settings.dataset_id}' "
                f"(attempt {attempt}/{_MAX_RETRIES})... "
                f"HF cache: {_HF_CACHE}"
            )

            # Ensure cache directory exists
            os.makedirs(_HF_CACHE, exist_ok=True)

            ds = datasets.load_dataset(
                settings.dataset_id,
                split="train",
                trust_remote_code=True,
            )
            df = ds.to_pandas()
            logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")
            return df

        except Exception as exc:
            last_error = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                f"Attempt {attempt} failed: {type(exc).__name__}: {exc}. "
                f"Retrying in {wait}s..."
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    raise DataSourceError(
        f"Failed to load dataset after {_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
