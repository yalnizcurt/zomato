"""
Phase 2 — Input Collector

Collect user preferences via CLI (extensible to Web/API).

Functions:
    collect_cli(available_cities, available_cuisines) -> dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def collect_cli(
    available_cities: list[str],
    available_cuisines: list[str],
) -> dict:
    """
    Collect user preferences interactively via the terminal.

    Args:
        available_cities: List of valid city names from the dataset.
        available_cuisines: List of valid cuisine names from the dataset.

    Returns:
        Raw input dictionary (not yet validated).
    """
    print("\n" + "=" * 60)
    print("  🍽️  Restaurant Recommendation System")
    print("=" * 60)
    print()

    # Location
    print(f"📍 Available cities: {', '.join(available_cities[:15])}...")
    location = input("   Enter your city: ").strip()

    # Budget
    print("\n💰 Budget options: low (< ₹500), medium (₹500–1500), high (> ₹1500)")
    budget = input("   Enter your budget [low/medium/high]: ").strip().lower()

    # Cuisine
    print(f"\n🍕 Popular cuisines: {', '.join(available_cuisines[:15])}...")
    cuisine_input = input("   Enter preferred cuisines (comma-separated): ").strip()
    cuisines = [c.strip().lower() for c in cuisine_input.split(",") if c.strip()]

    # Rating
    rating_input = input("\n⭐ Minimum rating [0.0–5.0, default 3.0]: ").strip()
    min_rating = float(rating_input) if rating_input else 3.0

    # Additional
    additional = input(
        "\n💬 Any other preferences? (e.g., family-friendly, rooftop): "
    ).strip()

    raw_input = {
        "location": location,
        "budget": budget,
        "cuisines": cuisines,
        "min_rating": min_rating,
        "additional_preferences": additional,
    }

    logger.info(f"Raw user input collected: {raw_input}")
    return raw_input
