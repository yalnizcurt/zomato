"""
Restaurant Recommender — Flask Web Application

Serves the web UI and handles recommendation API requests.
Replaces CLI-based input collection with a beautiful web interface.

Usage:
    python app.py
"""

from __future__ import annotations

import logging
import sys
import json
import time
import threading

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

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

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Module-level cache for the cleaned DataFrame
_df_cache = None
_available_cities = []
_available_cuisines = []


# ---------------------------------------------------------------------------
# Server-side rate limiter for the recommend endpoint
# ---------------------------------------------------------------------------
class _EndpointThrottle:
    """
    Simple per-endpoint throttle: allows at most `max_calls` within
    `window_seconds`. Prevents rapid-fire submissions from burning
    through the Gemini free-tier RPM quota.
    """

    def __init__(self, max_calls: int = 3, window_seconds: float = 30.0):
        self._max = max_calls
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def is_allowed(self) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        with self._lock:
            now = time.time()
            self._timestamps = [
                ts for ts in self._timestamps if now - ts < self._window
            ]
            if len(self._timestamps) >= self._max:
                retry_after = self._window - (now - self._timestamps[0])
                return False, max(retry_after, 1.0)
            self._timestamps.append(now)
            return True, 0.0


_recommend_throttle = _EndpointThrottle(max_calls=3, window_seconds=30.0)


def _ensure_data_loaded():
    """Load and cache the dataset on first request (Phase 1)."""
    global _df_cache, _available_cities, _available_cuisines

    if _df_cache is not None:
        return

    logger.info("━━━ Phase 1: Data Ingestion & Preprocessing ━━━")
    from data_ingestion.data_store import get_dataframe

    _df_cache = get_dataframe()

    _available_cities = sorted(_df_cache["city"].unique().tolist())
    all_cuisines = set()
    for cuisine_list in _df_cache["cuisines"]:
        if isinstance(cuisine_list, list):
            all_cuisines.update(cuisine_list)
    _available_cuisines = sorted(all_cuisines)

    logger.info(
        f"Data loaded: {len(_df_cache)} restaurants, "
        f"{len(_available_cities)} cities, {len(_available_cuisines)} cuisines."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main web UI."""
    return send_from_directory("static", "index.html")


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return available cities and cuisines for the UI dropdowns."""
    try:
        _ensure_data_loaded()
        return jsonify({
            "cities": _available_cities,
            "cuisines": _available_cuisines,
            "budget_levels": ["low", "medium", "high"],
        })
    except (DataSourceError, SchemaError, DataQualityError) as exc:
        logger.error(f"Data loading failed: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Accept user preferences via JSON POST and return recommendations.

    Expected JSON body:
    {
        "location": "bangalore",
        "budget": "medium",
        "cuisines": ["italian", "chinese"],
        "min_rating": 3.5,
        "additional_preferences": "family-friendly"
    }
    """
    # --- Server-side throttle ---
    allowed, retry_after = _recommend_throttle.is_allowed()
    if not allowed:
        logger.warning(f"Throttled request — retry after {retry_after:.0f}s")
        resp = jsonify({
            "success": False,
            "error": f"Too many requests. Please wait {int(retry_after)} seconds before trying again.",
            "retry_after": round(retry_after),
        })
        resp.status_code = 429
        resp.headers["Retry-After"] = str(int(retry_after))
        return resp

    try:
        _ensure_data_loaded()

        # --- Phase 2: Validate input ---
        raw_input = request.get_json(force=True)
        logger.info(f"━━━ Phase 2: Web Input Received ━━━ {raw_input}")

        # Parse cuisines if sent as string
        cuisines = raw_input.get("cuisines", [])
        if isinstance(cuisines, str):
            cuisines = [c.strip().lower() for c in cuisines.split(",") if c.strip()]
        raw_input["cuisines"] = cuisines

        # Ensure min_rating is float
        try:
            raw_input["min_rating"] = float(raw_input.get("min_rating", 3.0))
        except (TypeError, ValueError):
            raw_input["min_rating"] = 3.0

        from user_input.input_validator import validate
        preferences = validate(raw_input, _available_cities, _available_cuisines)

        # --- Phase 3: Filter + Prompt ---
        logger.info("━━━ Phase 3: Integration Layer ━━━")
        from filtering.filter_engine import filter_restaurants
        from filtering.shortlister import shortlist
        from filtering.prompt_builder import build_prompt

        filtered_df = filter_restaurants(_df_cache, preferences)
        candidates_df = shortlist(filtered_df, n=settings.max_candidates)
        prompt = build_prompt(preferences, candidates_df)

        # --- Phase 4: LLM ---
        logger.info("━━━ Phase 4: Recommendation Engine ━━━")
        from llm.llm_client import call_llm
        from llm.response_parser import parse
        from llm.fallback import heuristic_rank

        try:
            raw_response = call_llm(prompt)
            recommendations = parse(raw_response, candidates_df)
        except (LLMError, LLMParseError) as exc:
            logger.warning(f"LLM failed ({exc}), using heuristic fallback.")
            recommendations = heuristic_rank(
                candidates_df, n=settings.max_recommendations
            )

        # --- Phase 5: Format for web ---
        from output.formatter import format_for_web
        results = format_for_web(recommendations)

        return jsonify({
            "success": True,
            "count": len(results),
            "recommendations": results,
            "filters_applied": {
                "location": preferences.location,
                "budget": preferences.budget.value,
                "cuisines": preferences.cuisines,
                "min_rating": preferences.min_rating,
            },
        })

    except InputValidationError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "field": exc.field_name,
            "suggestions": exc.suggestions,
        }), 400

    except EmptyFilterResultError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "relaxation_steps": exc.relaxation_steps,
        }), 404

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(exc)}",
        }), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🍽️  Restaurant Recommender — Web UI")
    print(f"   Environment: {settings.env.value}")
    print(f"   LLM Provider: {settings.llm_provider.value}")
    print(f"   Open http://localhost:5001 in your browser\n")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=(settings.env.value == "development"),
    )
