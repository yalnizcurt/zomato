"""
Restaurant Recommender — Shared Data Models

All data contracts between phases are defined here so that every module
references the same source of truth.

Models:
    - BudgetLevel       : Enum for user budget tiers
    - UserPreferences   : Phase 2 output — validated user input
    - Recommendation    : Phase 4 output — a single ranked restaurant
    - RestaurantRecord  : Phase 1 output — a single cleaned row from the dataset

Custom Exceptions:
    - DataSourceError   : Phase 1 — dataset cannot be loaded
    - SchemaError       : Phase 1 — dataset schema doesn't match expectations
    - DataQualityError  : Phase 1 — data quality below threshold
    - ValidationError   : Phase 2 — user input fails validation
    - PromptBuildError  : Phase 3 — prompt template has unresolved variables
    - LLMError          : Phase 4 — LLM API call failed
    - ParseError        : Phase 4 — LLM response cannot be parsed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ===================================================================
# Enums
# ===================================================================

class BudgetLevel(Enum):
    """User budget tiers with human-readable labels."""
    LOW = "low"         # < ₹500
    MEDIUM = "medium"   # ₹500 – ₹1500
    HIGH = "high"       # > ₹1500

    def __str__(self) -> str:
        return self.value


# ===================================================================
# Data Models
# ===================================================================

@dataclass(frozen=True)
class RestaurantRecord:
    """
    A single cleaned restaurant row from the Zomato dataset.
    Output of Phase 1 (Data Ingestion).
    """
    restaurant_name: str
    city: str
    cuisines: list[str]
    average_cost_for_two: int
    aggregate_rating: float
    votes: int
    has_online_delivery: bool

    def cuisines_display(self) -> str:
        """Comma-separated cuisine string for display."""
        return ", ".join(self.cuisines)


@dataclass(frozen=True)
class UserPreferences:
    """
    Validated user preferences.
    Output of Phase 2 (User Input Collection).
    """
    location: str                              # Lowercased, e.g. "bangalore"
    budget: BudgetLevel                        # Enum value
    cuisines: list[str] = field(default_factory=list)  # e.g. ["italian", "chinese"]
    min_rating: float = 3.0                    # 0.0 – 5.0
    additional_preferences: str = ""           # Free-text, max 500 chars

    def __post_init__(self) -> None:
        # Validate rating range (defensive — Phase 2 validator should catch this)
        if not (0.0 <= self.min_rating <= 5.0):
            raise ValueError(
                f"min_rating must be between 0.0 and 5.0, got {self.min_rating}"
            )
        # Validate additional_preferences length
        if len(self.additional_preferences) > 500:
            # Truncate rather than reject at model level (validator warns)
            object.__setattr__(
                self, "additional_preferences",
                self.additional_preferences[:500]
            )


@dataclass
class Recommendation:
    """
    A single LLM-ranked restaurant recommendation.
    Output of Phase 4 (Recommendation Engine).
    """
    rank: int                                  # 1-indexed position
    restaurant_name: str                       # From candidate data
    cuisines: list[str]                        # Cuisine types
    rating: float                              # Aggregate rating (0.0 – 5.0)
    cost_for_two: int                          # Average cost in INR
    location: str                              # City / area
    explanation: str                           # LLM-generated: why this fits
    trade_offs: Optional[str] = None           # Optional caveats

    def cuisines_display(self) -> str:
        """Comma-separated cuisine string for display."""
        return ", ".join(self.cuisines)


# ===================================================================
# Custom Exceptions — organized by phase
# ===================================================================

# --- Phase 1: Data Ingestion ---

class DataSourceError(Exception):
    """Raised when the dataset cannot be loaded from any source."""
    pass


class SchemaError(Exception):
    """Raised when the dataset schema doesn't match expectations."""

    def __init__(self, missing_columns: list[str], extra_columns: list[str] | None = None):
        self.missing_columns = missing_columns
        self.extra_columns = extra_columns or []
        parts = [f"Missing columns: {missing_columns}"]
        if self.extra_columns:
            parts.append(f"Unexpected columns: {extra_columns}")
        super().__init__(" | ".join(parts))


class DataQualityError(Exception):
    """Raised when data quality is below acceptable thresholds."""

    def __init__(self, column: str, null_percentage: float):
        self.column = column
        self.null_percentage = null_percentage
        super().__init__(
            f"Column '{column}' has {null_percentage:.1f}% null values "
            f"(threshold: 30%)"
        )


# --- Phase 2: User Input ---

class InputValidationError(Exception):
    """Raised when user input fails validation."""

    def __init__(self, field_name: str, message: str, suggestions: list[str] | None = None):
        self.field_name = field_name
        self.suggestions = suggestions or []
        detail = f"Invalid '{field_name}': {message}"
        if self.suggestions:
            detail += f" — Did you mean: {', '.join(self.suggestions)}?"
        super().__init__(detail)


# --- Phase 3: Integration Layer ---

class PromptBuildError(Exception):
    """Raised when a prompt template has unresolved variables."""

    def __init__(self, unresolved_vars: list[str]):
        self.unresolved_vars = unresolved_vars
        super().__init__(
            f"Prompt template has unresolved variables: {unresolved_vars}"
        )


class EmptyFilterResultError(Exception):
    """Raised when all filters yield zero candidates."""

    def __init__(self, relaxation_steps: list[str] | None = None):
        self.relaxation_steps = relaxation_steps or []
        msg = "No restaurants match the given criteria."
        if self.relaxation_steps:
            msg += f" Tried relaxing: {', '.join(self.relaxation_steps)}"
        super().__init__(msg)


# --- Phase 4: LLM Engine ---

class LLMError(Exception):
    """Raised when the LLM API call fails after all retries."""

    def __init__(self, provider: str, status_code: int | None, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")


class LLMParseError(Exception):
    """Raised when the LLM response cannot be parsed into Recommendations."""

    def __init__(self, raw_response: str, reason: str):
        self.raw_response = raw_response[:500]  # Truncate for safety
        self.reason = reason
        super().__init__(f"Failed to parse LLM response: {reason}")
