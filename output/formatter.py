"""
Phase 5 — Output Formatter

Convert Recommendation objects to display-ready structures.
"""

from __future__ import annotations

from models import Recommendation


def format_for_cli(recommendations: list[Recommendation]) -> list[dict]:
    """Format recommendations for CLI card rendering."""
    return [
        {
            "rank": rec.rank,
            "name": rec.restaurant_name,
            "cuisines": rec.cuisines_display(),
            "rating": f"{rec.rating}/5",
            "cost": f"₹{rec.cost_for_two:,}",
            "location": rec.location.title(),
            "explanation": rec.explanation,
            "trade_offs": rec.trade_offs,
        }
        for rec in recommendations
    ]


def format_for_web(recommendations: list[Recommendation]) -> list[dict]:
    """Format recommendations for web rendering."""
    return [
        {
            "rank": rec.rank,
            "restaurant_name": rec.restaurant_name,
            "cuisines": rec.cuisines,
            "rating": rec.rating,
            "cost_for_two": rec.cost_for_two,
            "location": rec.location,
            "explanation": rec.explanation,
            "trade_offs": rec.trade_offs,
        }
        for rec in recommendations
    ]


def to_json(recommendations: list[Recommendation]) -> list[dict]:
    """Convert recommendations to JSON-serializable dicts."""
    return format_for_web(recommendations)
