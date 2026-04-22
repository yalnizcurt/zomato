"""
Phase 4 — LLM Client

Abstract LLM provider with:
  - Singleton client instances (avoids recreating connections per call)
  - Token-bucket rate limiter (proactively prevents 429s)
  - Exponential backoff with jitter for transient errors
  - Fast-fail to heuristic fallback after consecutive rate-limit hits
"""

from __future__ import annotations

import logging
import random
import threading
import time

from config import settings, LLMProvider, OpenAIConfig, GeminiConfig
from models import LLMError

logger = logging.getLogger(__name__)


# ===================================================================
# Rate Limiter — token-bucket per provider
# ===================================================================

class _RateLimiter:
    """
    Simple token-bucket rate limiter.

    Gemini free tier allows ~15 RPM (requests per minute).
    We default to 10 RPM to stay safely under the limit.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
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


# Global rate limiters — one per provider
_gemini_limiter = _RateLimiter(max_requests=10, window_seconds=60.0)
_openai_limiter = _RateLimiter(max_requests=50, window_seconds=60.0)


# ===================================================================
# Singleton clients
# ===================================================================

_gemini_client = None
_openai_client = None
_client_lock = threading.Lock()


def _get_gemini_client(config: GeminiConfig):
    """Return a singleton Gemini client instance."""
    global _gemini_client
    with _client_lock:
        if _gemini_client is None:
            from google import genai
            _gemini_client = genai.Client(api_key=config.api_key)
            logger.info("Gemini client initialised (singleton).")
        return _gemini_client


def _get_openai_client(config: OpenAIConfig):
    """Return a singleton OpenAI client instance."""
    global _openai_client
    with _client_lock:
        if _openai_client is None:
            import openai
            _openai_client = openai.OpenAI(api_key=config.api_key)
            logger.info("OpenAI client initialised (singleton).")
        return _openai_client


# ===================================================================
# Public entry point
# ===================================================================

def call_llm(prompt: str) -> str:
    """Send prompt to configured LLM, return raw response text."""
    config = settings.active_llm_config

    if settings.llm_provider == LLMProvider.OPENAI:
        limiter = _openai_limiter
    else:
        limiter = _gemini_limiter

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

    if settings.llm_provider == LLMProvider.OPENAI:
        return _call_openai(prompt, config, limiter)
    return _call_gemini(prompt, config, limiter)


# ===================================================================
# Provider implementations
# ===================================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if the exception is a rate-limit / quota error."""
    msg = str(exc).lower()
    return any(indicator in msg for indicator in [
        "429", "resource_exhausted", "rate limit",
        "quota", "too many requests",
    ])


def _call_openai(prompt: str, config: OpenAIConfig, limiter: _RateLimiter) -> str:
    client = _get_openai_client(config)
    last_error = None

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"OpenAI call attempt {attempt}/{config.max_retries}")
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
            limiter.record_success()
            return response.choices[0].message.content

        except Exception as exc:
            last_error = exc
            if _is_rate_limit_error(exc):
                limiter.record_429()
            if attempt < config.max_retries:
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                logger.warning(f"OpenAI error: {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

    raise LLMError("openai", None, f"All retries failed: {last_error}")


def _call_gemini(prompt: str, config: GeminiConfig, limiter: _RateLimiter) -> str:
    client = _get_gemini_client(config)
    last_error = None
    max_retries = config.max_retries  # Use configured retries (default 2)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Gemini call attempt {attempt}/{max_retries}")
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
                config={
                    "temperature": config.temperature,
                    "max_output_tokens": max(config.max_tokens, 8192),
                    "response_mime_type": "application/json",
                    # Thinking models (gemini-flash-latest → gemini-3-flash)
                    # share max_output_tokens between thinking + output.
                    # Cap thinking to 1024 so the JSON response isn't truncated.
                    "thinking_config": {"thinking_budget": 1024},
                },
            )

            # Log token usage for debugging
            meta = response.usage_metadata
            thinking_tokens = getattr(meta, "thoughts_token_count", 0) or 0
            logger.info(
                f"Gemini tokens — prompt: {meta.prompt_token_count}, "
                f"output: {meta.candidates_token_count}, "
                f"thinking: {thinking_tokens}, "
                f"total: {meta.total_token_count}, "
                f"finish: {response.candidates[0].finish_reason}"
            )

            result_text = response.text
            logger.debug(f"Gemini raw response ({len(result_text)} chars): {result_text[:500]}")
            limiter.record_success()
            return result_text

        except Exception as exc:
            last_error = exc
            is_rate_limit = _is_rate_limit_error(exc)

            if is_rate_limit:
                limiter.record_429()
                if attempt < max_retries:
                    # Moderate backoff: 5s, 10s + jitter
                    wait = (5 * attempt) + random.uniform(1, 3)
                    logger.warning(
                        f"Rate limited (429). Waiting {wait:.0f}s before "
                        f"retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                else:
                    # Don't retry further — let it fall through to heuristic
                    logger.warning(
                        f"Rate limited on final attempt. "
                        f"Will fall back to heuristic ranking."
                    )
            elif attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"Gemini error: {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

    raise LLMError("gemini", None, f"All retries failed: {last_error}")
