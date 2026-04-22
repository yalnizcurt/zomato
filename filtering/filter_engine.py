"""
Phase 3 — Filter Engine

Apply structured filters to the clean DataFrame based on UserPreferences.

Functions:
    filter_restaurants(df, preferences) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd

from config import settings
from models import UserPreferences

logger = logging.getLogger(__name__)


def filter_restaurants(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """
    Filter the restaurant DataFrame using structured user preferences.

    Applies filters in order:
        1. Location (exact match)
        2. Budget range
        3. Minimum rating
        4. Cuisine overlap (OR logic)

    If any filter yields zero results, it is progressively relaxed.

    Args:
        df: Clean restaurant DataFrame.
        prefs: Validated user preferences.

    Returns:
        Filtered DataFrame (may be smaller or equal to input).
    """
    result = df.copy()
    relaxations: list[str] = []

    # --- Filter 1: Location ---
    location_mask = result["city"] == prefs.location
    if location_mask.any():
        result = result[location_mask]
        logger.info(f"Location filter '{prefs.location}': {len(result)} restaurants.")
    else:
        logger.warning(f"No restaurants in '{prefs.location}' — skipping location filter.")
        relaxations.append("location")

    # --- Filter 2: Budget ---
    budget_range = settings.budget_ranges.get(prefs.budget.value, (0, 100_000))
    low, high = budget_range
    budget_mask = (result["average_cost_for_two"] >= low) & (result["average_cost_for_two"] <= high)
    budget_filtered = result[budget_mask]
    if len(budget_filtered) > 0:
        result = budget_filtered
        logger.info(f"Budget filter '{prefs.budget.value}' (₹{low}–₹{high}): {len(result)} restaurants.")
    else:
        logger.warning(f"No restaurants in budget range — skipping budget filter.")
        relaxations.append("budget")

    # --- Filter 3: Minimum rating ---
    rating_mask = result["aggregate_rating"] >= prefs.min_rating
    rating_filtered = result[rating_mask]
    if len(rating_filtered) > 0:
        result = rating_filtered
        logger.info(f"Rating filter (>= {prefs.min_rating}): {len(result)} restaurants.")
    else:
        logger.warning(f"No restaurants with rating >= {prefs.min_rating} — skipping rating filter.")
        relaxations.append("min_rating")

    # --- Filter 4: Cuisine (OR logic) ---
    if prefs.cuisines:
        cuisine_mask = result["cuisines"].apply(
            lambda cs: any(c in cs for c in prefs.cuisines)
        )
        cuisine_filtered = result[cuisine_mask]
        if len(cuisine_filtered) > 0:
            result = cuisine_filtered
            logger.info(f"Cuisine filter {prefs.cuisines}: {len(result)} restaurants.")
        else:
            logger.warning(f"No cuisine matches — skipping cuisine filter.")
            relaxations.append("cuisines")

    if relaxations:
        logger.info(f"Relaxed filters: {relaxations}")

    logger.info(f"Final filtered count: {len(result)} restaurants.")
    return result
