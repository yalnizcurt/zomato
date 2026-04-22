"""
Restaurant Recommender — Application Configuration

Centralizes all configuration: environment detection, LLM provider settings,
data source paths, and application-wide defaults. Every phase reads from
this single source of truth.

Usage:
    from config import settings
    print(settings.llm_provider)
    print(settings.max_candidates)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (if present) — never crashes if missing
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Environment(Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


# ---------------------------------------------------------------------------
# LLM-specific configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI provider settings."""
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_seconds: int = 15
    max_retries: int = 2


@dataclass(frozen=True)
class GeminiConfig:
    """Google Gemini provider settings."""
    api_key: str
    model: str = "gemini-flash-latest"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: int = 30
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Budget range mapping (INR)
# ---------------------------------------------------------------------------
BUDGET_RANGES: dict[str, tuple[int, int]] = {
    "low":    (0,    500),
    "medium": (500,  1500),
    "high":   (1500, 100_000),
}


# ---------------------------------------------------------------------------
# City alias map — normalizes common alternate spellings
# ---------------------------------------------------------------------------
CITY_ALIASES: dict[str, str] = {
    "bengaluru": "bangalore",
    "bombay":    "mumbai",
    "calcutta":  "kolkata",
    "madras":    "chennai",
    "poona":     "pune",
    "trivandrum":"thiruvananthapuram",
}


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
DATASET_CACHE_DIR = _PROJECT_ROOT / "data_cache"

# Schema that Phase 1 must guarantee after cleaning
EXPECTED_SCHEMA: dict[str, type] = {
    "restaurant_name":       str,
    "city":                  str,
    "cuisines":              list,
    "average_cost_for_two":  int,
    "aggregate_rating":      float,
    "votes":                 int,
    "has_online_delivery":   bool,
}


# ---------------------------------------------------------------------------
# Main Settings dataclass
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """
    Application-wide settings, populated from environment variables
    with sensible defaults for development.
    """

    # --- Environment ---
    env: Environment = field(default_factory=lambda: Environment(
        os.getenv("APP_ENV", "development").lower()
    ))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- LLM Provider ---
    llm_provider: LLMProvider = field(default_factory=lambda: LLMProvider(
        os.getenv("LLM_PROVIDER", "gemini").lower()
    ))

    # --- Recommendation defaults ---
    max_candidates: int = field(default_factory=lambda: int(
        os.getenv("MAX_CANDIDATES", "15")
    ))
    max_recommendations: int = field(default_factory=lambda: int(
        os.getenv("MAX_RECOMMENDATIONS", "5")
    ))

    # --- Data ---
    dataset_id: str = DATASET_ID
    dataset_cache_dir: Path = DATASET_CACHE_DIR
    expected_schema: dict[str, type] = field(default_factory=lambda: EXPECTED_SCHEMA.copy())
    budget_ranges: dict[str, tuple[int, int]] = field(default_factory=lambda: BUDGET_RANGES.copy())
    city_aliases: dict[str, str] = field(default_factory=lambda: CITY_ALIASES.copy())

    # --- Derived LLM configs (populated in __post_init__) ---
    openai_config: Optional[OpenAIConfig] = field(default=None, init=False)
    gemini_config: Optional[GeminiConfig] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Build provider-specific configs from environment variables."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

        # Environment-specific defaults
        is_prod = self.env == Environment.PRODUCTION
        default_temp = 0.5 if is_prod else 0.3
        default_tokens = 4096 if is_prod else 2048
        default_timeout = 30 if is_prod else 30
        default_retries = 3 if is_prod else 2

        # OpenAI config
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and openai_key != "sk-your-openai-key-here":
            self.openai_config = OpenAIConfig(
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=default_temp,
                max_tokens=default_tokens,
                timeout_seconds=default_timeout,
                max_retries=default_retries,
            )

        # Gemini config
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key and gemini_key != "your-gemini-key-here":
            self.gemini_config = GeminiConfig(
                api_key=gemini_key,
                model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
                temperature=default_temp,
                max_tokens=default_tokens,
                timeout_seconds=default_timeout,
                max_retries=default_retries,
            )

    @property
    def active_llm_config(self) -> OpenAIConfig | GeminiConfig:
        """Return the config for the currently selected LLM provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            if self.openai_config is None:
                raise ValueError(
                    "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                    "Check your .env file."
                )
            return self.openai_config
        else:
            if self.gemini_config is None:
                raise ValueError(
                    "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. "
                    "Check your .env file."
                )
            return self.gemini_config

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
