"""
Phase 3 — Prompt Builder

Construct the LLM prompt from user preferences and candidate data.

Functions:
    build_prompt(prefs, candidates_df) -> str
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from models import UserPreferences, PromptBuildError
from filtering.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_V1

logger = logging.getLogger(__name__)


def build_prompt(prefs: UserPreferences, candidates_df: pd.DataFrame) -> str:
    """
    Build the full LLM prompt by combining system prompt, user context,
    and candidate data.

    Args:
        prefs: Validated user preferences.
        candidates_df: Shortlisted candidates DataFrame.

    Returns:
        Complete prompt string ready to send to the LLM.

    Raises:
        PromptBuildError: If any template variables remain unresolved.
    """
    # Format candidate data as a numbered list
    candidates_text = _format_candidates(candidates_df)

    # Build the user prompt from template
    user_prompt = USER_PROMPT_V1.format(
        location=prefs.location.title(),
        budget=prefs.budget.value,
        cuisines=", ".join(prefs.cuisines) if prefs.cuisines else "any",
        min_rating=prefs.min_rating,
        additional_preferences=prefs.additional_preferences or "none",
        candidates=candidates_text,
        max_recommendations=5,
    )

    # Combine system + user prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # Validate: no unresolved {variables}
    unresolved = re.findall(r"\{(\w+)\}", full_prompt)
    if unresolved:
        raise PromptBuildError(unresolved_vars=unresolved)

    logger.info(f"Prompt built: {len(full_prompt)} characters.")
    return full_prompt


def _format_candidates(df: pd.DataFrame) -> str:
    """Format candidate rows into a numbered text list for the prompt."""
    lines = []
    for i, row in df.iterrows():
        cuisines = row["cuisines"]
        if isinstance(cuisines, list):
            cuisines = ", ".join(cuisines)

        lines.append(
            f"{len(lines) + 1}. "
            f"Name: {row['restaurant_name']} | "
            f"Cuisines: {cuisines} | "
            f"Rating: {row['aggregate_rating']}/5 | "
            f"Cost for two: ₹{row['average_cost_for_two']} | "
            f"City: {row['city']} | "
            f"Votes: {row['votes']} | "
            f"Online delivery: {'Yes' if row.get('has_online_delivery') else 'No'}"
        )
    return "\n".join(lines)
