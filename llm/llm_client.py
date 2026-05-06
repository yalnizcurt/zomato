"""
Phase 4 — LLM Client

Abstract LLM provider with:
  - Singleton client instances (avoids recreating connections per call)
  - Token-bucket rate limiter (proactively prevents 429s)
  - Exponential backoff with jitter for transient errors
  - Fast-fail to heuristic fallback after consecutive rate-limit hits

Uses Groq API (OpenAI-compatible) with llama-3.3-70b-versatile.
"""

from __future__ import annotations

import logging
import random
import threading
import time

from config import settings, LLMProvider, GroqConfig
from models import LLMError

logger = logging.getLogger(__name__)


# ===================================================================
# Rate Limiter — token-bucket per provider
# ===================================================================

class _RateLimiter:
    """
    Simple token-bucket rate limiter.

    Groq free tier allows ~30 RPM (requests per minute).
    We default to 25 RPM to stay safely under the limit.
    """

    def __init__(self, max_requests: int = 25, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()
        self._consecutive_429s = 0

    def wait_if_needed(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.time()
            # Purge timestamps older than the window
            self._timestamps = [
                ts for ts in self._timestamps
                if now - ts < self._window
            ]

            if len(self._timestamps) >= self._max:
                # Wait until the oldest timestamp exits the window
                sleep_time = self._window - (now - self._timestamps[0]) + 0.5
                logger.info(
                    f"Rate limiter: throttling for {sleep_time:.1f}s "
                    f"({len(self._timestamps)}/{self._max} slots used)"
                )
                time.sleep(max(sleep_time, 1.0))

            self._timestamps.append(time.time())

    def record_429(self) -> None:
        """Track consecutive 429 errors."""
        with self._lock:
            self._consecutive_429s += 1

    def record_success(self) -> None:
        """Reset 429 counter on success."""
        with self._lock:
            self._consecutive_429s = 0

    @property
    def should_fast_fail(self) -> bool:
        """After 2 consecutive 429s, skip LLM and go straight to fallback."""
        with self._lock:
            return self._consecutive_429s >= 2


# Global rate limiter for Groq
_groq_limiter = _RateLimiter(max_requests=25, window_seconds=60.0)


# ===================================================================
# Singleton clients
# ===================================================================

_groq_client = None
_client_lock = threading.Lock()


def _get_groq_client(config: GroqConfig):
    """Return a singleton Groq client instance (OpenAI-compatible)."""
    global _groq_client
    with _client_lock:
        if _groq_client is None:
            import openai
            _groq_client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            logger.info("Groq client initialised (singleton, OpenAI-compatible).")
        return _groq_client


# ===================================================================
# Public entry point
# ===================================================================

def call_llm(prompt: str) -> str:
    """Send prompt to configured LLM, return raw response text."""
    config = settings.active_llm_config
    limiter = _groq_limiter

    # Fast-fail: if we've been rate-limited repeatedly, don't even try
    if limiter.should_fast_fail:
        logger.warning(
            "Multiple consecutive 429 errors detected — "
            "skipping LLM call to trigger heuristic fallback immediately."
        )
        raise LLMError(
            settings.llm_provider.value, 429,
            "Rate limit exceeded repeatedly. Using heuristic fallback."
        )

    # Proactive throttle
    limiter.wait_if_needed()

    return _call_groq(prompt, config, limiter)


# ===================================================================
# Provider implementation — Groq (OpenAI-compatible)
# ===================================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if the exception is a rate-limit / quota error."""
    msg = str(exc).lower()
    return any(indicator in msg for indicator in [
        "429", "resource_exhausted", "rate limit",
        "quota", "too many requests",
    ])


def _call_groq(prompt: str, config: GroqConfig, limiter: _RateLimiter) -> str:
    client = _get_groq_client(config)
    last_error = None

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"Groq call attempt {attempt}/{config.max_retries}")
            parts = prompt.split("\n\n", 1)
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": parts[0]},
                    {"role": "user", "content": parts[1] if len(parts) > 1 else prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
            )

            result_text = response.choices[0].message.content
            logger.info(
                f"Groq response — model: {config.model}, "
                f"tokens: {response.usage.total_tokens if response.usage else 'N/A'}, "
                f"chars: {len(result_text)}"
            )
            logger.debug(f"Groq raw response: {result_text[:500]}")
            limiter.record_success()
            return result_text

        except Exception as exc:
            last_error = exc
            is_rate_limit = _is_rate_limit_error(exc)

            if is_rate_limit:
                limiter.record_429()
                if attempt < config.max_retries:
                    wait = (5 * attempt) + random.uniform(1, 3)
                    logger.warning(
                        f"Rate limited (429). Waiting {wait:.0f}s before "
                        f"retry {attempt + 1}/{config.max_retries}..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"Rate limited on final attempt. "
                        f"Will fall back to heuristic ranking."
                    )
            elif attempt < config.max_retries:
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"Groq error: {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

    raise LLMError("groq", None, f"All retries failed: {last_error}")
