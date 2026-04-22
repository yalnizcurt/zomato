"""
Phase 3 — Candidate Shortlister

Rank and select the top N candidates from filtered results.

Functions:
    shortlist(df, n) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from models import EmptyFilterResultError

logger = logging.getLogger(__name__)


def shortlist(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Rank candidates by a composite score and return the top N.

    Composite score = aggregate_rating × log2(votes + 1)
    This balances high ratings with popularity.

    Args:
        df: Filtered restaurant DataFrame.
        n: Number of candidates to return.

    Returns:
        Top N candidates sorted by composite score (descending).

    Raises:
        EmptyFilterResultError: If the input DataFrame is empty.
    """
    if df.empty:
        raise EmptyFilterResultError(
            relaxation_steps=["All filters were already relaxed."]
        )

    # Composite score: rating × log(votes + 1)
    df = df.copy()
    df["_score"] = df["aggregate_rating"] * df["votes"].apply(
        lambda v: math.log2(max(v, 1) + 1)
    )

    # Sort descending and take top N
    df = df.sort_values("_score", ascending=False).head(n)

    # Clean up internal column
    df = df.drop(columns=["_score"])

    logger.info(f"Shortlisted top {len(df)} candidates.")
    return df.reset_index(drop=True)
