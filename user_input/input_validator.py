"""
Phase 2 — Input Validator

Validate and sanitize raw user input into a UserPreferences object.

Functions:
    validate(raw_input, available_cities, available_cuisines) -> UserPreferences
"""

from __future__ import annotations

import logging
import re

from thefuzz import process as fuzz_process

from models import BudgetLevel, UserPreferences, InputValidationError

logger = logging.getLogger(__name__)

# Maximum allowed length for free-text preferences
_MAX_ADDITIONAL_LEN = 500

# Minimum fuzzy match score to accept a suggestion
_FUZZY_THRESHOLD = 70


def validate(
    raw_input: dict,
    available_cities: list[str],
    available_cuisines: list[str],
) -> UserPreferences:
    """
    Validate raw user input and return a UserPreferences object.

    Args:
        raw_input: Dictionary from input_collector.
        available_cities: Valid city names from the dataset.
        available_cuisines: Valid cuisine names from the dataset.

    Returns:
        Validated UserPreferences.

    Raises:
        InputValidationError: If any field fails validation.
    """
    location = _validate_location(raw_input.get("location", ""), available_cities)
    budget = _validate_budget(raw_input.get("budget", "medium"))
    cuisines = _validate_cuisines(raw_input.get("cuisines", []), available_cuisines)
    min_rating = _validate_rating(raw_input.get("min_rating", 3.0))
    additional = _sanitize_text(raw_input.get("additional_preferences", ""))

    prefs = UserPreferences(
        location=location,
        budget=budget,
        cuisines=cuisines,
        min_rating=min_rating,
        additional_preferences=additional,
    )

    logger.info(f"Validated preferences: {prefs}")
    return prefs


def _validate_location(location: str, available_cities: list[str]) -> str:
    """Validate and fuzzy-match the location."""
    location = location.strip().lower()

    if not location:
        raise InputValidationError(
            field_name="location",
            message="Location cannot be empty.",
            suggestions=available_cities[:5],
        )

    # Sanitize: strip non-alphanumeric except spaces and hyphens
    location = re.sub(r"[^a-z0-9\s\-]", "", location)

    # Exact match
    if location in available_cities:
        return location

    # Fuzzy match
    match = fuzz_process.extractOne(location, available_cities)
    if match and match[1] >= _FUZZY_THRESHOLD:
        suggestion = match[0]
        logger.info(f"Fuzzy matched '{location}' → '{suggestion}' (score: {match[1]})")
        return suggestion

    raise InputValidationError(
        field_name="location",
        message=f"'{location}' not found in the dataset.",
        suggestions=available_cities[:5],
    )


def _validate_budget(budget: str) -> BudgetLevel:
    """Validate budget input against allowed enum values."""
    budget = budget.strip().lower()
    try:
        return BudgetLevel(budget)
    except ValueError:
        raise InputValidationError(
            field_name="budget",
            message=f"'{budget}' is not a valid budget level.",
            suggestions=["low", "medium", "high"],
        )


def _validate_cuisines(cuisines: list[str], available_cuisines: list[str]) -> list[str]:
    """Validate and fuzzy-match cuisines against available options."""
    if not cuisines:
        # Default: no cuisine filter (accept all)
        return []

    validated = []
    for cuisine in cuisines:
        cuisine = cuisine.strip().lower()
        if not cuisine:
            continue

        # Exact match
        if cuisine in available_cuisines:
            validated.append(cuisine)
            continue

        # Fuzzy match
        match = fuzz_process.extractOne(cuisine, available_cuisines)
        if match and match[1] >= _FUZZY_THRESHOLD:
            logger.info(f"Fuzzy matched cuisine '{cuisine}' → '{match[0]}' (score: {match[1]})")
            validated.append(match[0])
        else:
            logger.warning(f"Cuisine '{cuisine}' not found — skipping.")

    return validated


def _validate_rating(rating) -> float:
    """Validate and clamp the minimum rating."""
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        logger.warning(f"Invalid rating '{rating}', defaulting to 3.0")
        return 3.0

    return max(0.0, min(5.0, rating))


def _sanitize_text(text: str) -> str:
    """Sanitize free-text input: strip HTML, limit length."""
    # Remove HTML/script tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove common prompt injection patterns
    text = re.sub(r"(?i)(ignore|forget|disregard)\s+(all\s+)?(previous|above)\s+(instructions?|prompts?)", "", text)
    # Truncate
    if len(text) > _MAX_ADDITIONAL_LEN:
        logger.warning(f"Additional preferences truncated from {len(text)} to {_MAX_ADDITIONAL_LEN} chars.")
        text = text[:_MAX_ADDITIONAL_LEN]
    return text.strip()
