"""
Phase 4 — Fallback Handler

Heuristic ranking when the LLM fails.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from models import Recommendation

logger = logging.getLogger(__name__)


def heuristic_rank(candidates_df: pd.DataFrame, n: int = 5) -> list[Recommendation]:
    """
    Rank candidates by rating × log(votes) without using an LLM.
    Used as a fallback when the LLM is unavailable.
    """
    logger.info("Using heuristic fallback ranking (no LLM).")

    df = candidates_df.copy()
    df["_score"] = df["aggregate_rating"] * df["votes"].apply(
        lambda v: math.log2(max(v, 1) + 1)
    )
    df = df.sort_values("_score", ascending=False).head(n)

    recommendations = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        cuisines = row["cuisines"] if isinstance(row["cuisines"], list) else [row["cuisines"]]
        recommendations.append(Recommendation(
            rank=i,
            restaurant_name=row["restaurant_name"],
            cuisines=cuisines,
            rating=float(row["aggregate_rating"]),
            cost_for_two=int(row["average_cost_for_two"]),
            location=row["city"],
            explanation=(
                f"Ranked #{i} by our scoring algorithm based on its "
                f"{row['aggregate_rating']}/5 rating and {row['votes']} votes."
            ),
            trade_offs="AI-powered explanation unavailable — showing data-based ranking.",
        ))

    return recommendations
