"""
Restaurant Recommender — CLI Entry Point

Orchestrates all 5 phases in sequence via the terminal.
For the web UI, use: python app.py

    Phase 1: Data Ingestion & Preprocessing
    Phase 2: User Input Collection (CLI)
    Phase 3: Integration Layer (Filter + Prompt Engineering)
    Phase 4: Recommendation Engine (LLM)
    Phase 5: Output Display (Terminal)

Usage:
    python main.py          # CLI mode
    python app.py           # Web UI mode (recommended)
"""

from __future__ import annotations

import logging
import sys

from config import settings
from models import (
    DataSourceError,
    SchemaError,
    DataQualityError,
    InputValidationError,
    EmptyFilterResultError,
    LLMError,
    LLMParseError,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Phase orchestration
# ===================================================================

def run_phase1():
    """
    Phase 1 — Data Ingestion & Preprocessing
    Load, clean, and cache the Zomato dataset.
    Returns a clean, query-ready DataFrame.
    """
    logger.info("━━━ Phase 1: Data Ingestion & Preprocessing ━━━")

    from data_ingestion.data_loader import load_dataset
    from data_ingestion.data_cleaner import clean
    from data_ingestion.data_store import save, load, cache_exists

    # Use cached data if available
    if cache_exists():
        logger.info("Using cached dataset.")
        return load()

    # Load → Clean → Cache
    raw_df = load_dataset()
    clean_df = clean(raw_df)
    save(clean_df)
    logger.info(f"Phase 1 complete: {len(clean_df)} restaurants ready.")
    return clean_df


def run_phase2(available_cities: list[str], available_cuisines: list[str]):
    """
    Phase 2 — User Input Collection
    Collect and validate user preferences via CLI.
    Returns a UserPreferences object.
    """
    logger.info("━━━ Phase 2: User Input Collection ━━━")

    from user_input.input_collector import collect_cli
    from user_input.input_validator import validate

    raw_input = collect_cli(available_cities, available_cuisines)
    preferences = validate(raw_input, available_cities, available_cuisines)
    logger.info(f"Phase 2 complete: {preferences}")
    return preferences


def run_phase3(df, preferences):
    """
    Phase 3 — Integration Layer
    Filter candidates and build the LLM prompt.
    Returns (prompt_string, candidates_df).
    """
    logger.info("━━━ Phase 3: Integration Layer ━━━")

    from filtering.filter_engine import filter_restaurants
    from filtering.shortlister import shortlist
    from filtering.prompt_builder import build_prompt

    filtered_df = filter_restaurants(df, preferences)
    candidates_df = shortlist(filtered_df, n=settings.max_candidates)
    prompt = build_prompt(preferences, candidates_df)
    logger.info(
        f"Phase 3 complete: {len(candidates_df)} candidates → "
        f"prompt ({len(prompt)} chars)."
    )
    return prompt, candidates_df


def run_phase4(prompt, candidates_df):
    """
    Phase 4 — Recommendation Engine (LLM)
    Call the LLM, parse its response into Recommendation objects.
    Falls back to heuristic ranking on failure.
    """
    logger.info("━━━ Phase 4: Recommendation Engine (LLM) ━━━")

    from llm.llm_client import call_llm
    from llm.response_parser import parse
    from llm.fallback import heuristic_rank

    try:
        raw_response = call_llm(prompt)
        recommendations = parse(raw_response, candidates_df)
    except (LLMError, LLMParseError) as exc:
        logger.warning(f"LLM failed ({exc}), falling back to heuristic ranking.")
        recommendations = heuristic_rank(
            candidates_df, n=settings.max_recommendations
        )

    logger.info(f"Phase 4 complete: {len(recommendations)} recommendations.")
    return recommendations


def run_phase5(recommendations):
    """
    Phase 5 — Output Display
    Render recommendations to the terminal.
    """
    logger.info("━━━ Phase 5: Output Display ━━━")

    from output.formatter import format_for_cli
    from output.cli_renderer import render_cards

    formatted = format_for_cli(recommendations)
    render_cards(formatted)
    logger.info("Phase 5 complete: results displayed.")


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    """Run the full recommendation pipeline."""
    logger.info(
        f"🍽️  Restaurant Recommender starting "
        f"(env={settings.env.value}, llm={settings.llm_provider.value})"
    )

    try:
        # Phase 1 — Data
        df = run_phase1()

        # Extract available cities and cuisines for Phase 2 validation
        available_cities = sorted(df["city"].unique().tolist())
        all_cuisines = set()
        for cuisine_list in df["cuisines"]:
            if isinstance(cuisine_list, list):
                all_cuisines.update(cuisine_list)
        available_cuisines = sorted(all_cuisines)

        # Phase 2 — Input
        preferences = run_phase2(available_cities, available_cuisines)

        # Phase 3 — Filter + Prompt
        prompt, candidates_df = run_phase3(df, preferences)

        # Phase 4 — LLM
        recommendations = run_phase4(prompt, candidates_df)

        # Phase 5 — Display
        run_phase5(recommendations)

        return 0

    except DataSourceError as exc:
        logger.critical(f"Cannot load dataset: {exc}")
        print(f"\n❌ Fatal: {exc}")
        print("Please check your network connection or cached data.")
        return 1

    except SchemaError as exc:
        logger.critical(f"Dataset schema mismatch: {exc}")
        print(f"\n❌ Fatal: {exc}")
        return 1

    except DataQualityError as exc:
        logger.critical(f"Data quality issue: {exc}")
        print(f"\n❌ Fatal: {exc}")
        return 1

    except EmptyFilterResultError as exc:
        logger.warning(f"No results: {exc}")
        print(f"\n😔 {exc}")
        print("Try broadening your search criteria.")
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.")
        return 130

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        print(f"\n❌ Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
