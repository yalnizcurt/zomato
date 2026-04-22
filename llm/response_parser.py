"""
Phase 4 — Response Parser

Parse LLM raw response into Recommendation objects.
Handles edge cases from thinking models (gemini-flash-latest) that may
wrap JSON in markdown fences, add preamble text, or return partial matches.
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from models import Recommendation, LLMParseError

logger = logging.getLogger(__name__)


def parse(raw_response: str, candidates_df: pd.DataFrame) -> list[Recommendation]:
    """Parse LLM JSON response into validated Recommendation objects."""
    # Log the raw response for debugging (truncated)
    logger.debug(f"Raw LLM response ({len(raw_response)} chars): {raw_response[:500]}")

    data = _extract_json(raw_response)
    logger.debug(f"Extracted JSON type: {type(data).__name__}, keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")

    # Handle multiple possible response shapes:
    #   {"recommendations": [...]}
    #   [...]  (direct array)
    #   {"recommendations": {...}}  (single object instead of array)
    recs_data = data.get("recommendations", data if isinstance(data, list) else []) if isinstance(data, dict) else data

    # If the model returned a single dict instead of a list, wrap it
    if isinstance(recs_data, dict):
        recs_data = [recs_data]

    if not recs_data:
        raise LLMParseError(raw_response, "No recommendations found in response.")

    # Build set of valid candidate names for hallucination detection
    valid_names = set(candidates_df["restaurant_name"].str.lower().str.strip())

    # Also build a fuzzy lookup: strip punctuation for approximate matching
    clean_name_map = {}
    for name in candidates_df["restaurant_name"]:
        clean_key = re.sub(r"[^a-z0-9 ]", "", name.lower().strip())
        clean_name_map[clean_key] = name.lower().strip()

    recommendations = []
    seen_names = set()

    for item in recs_data:
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict recommendation item: {type(item)}")
            continue

        name = item.get("restaurant_name", "").strip()
        if not name:
            logger.warning("Skipping recommendation with empty restaurant_name.")
            continue

        name_lower = name.lower().strip()

        if name_lower in seen_names:
            logger.warning(f"Duplicate recommendation '{name}' — skipping.")
            continue

        # Check exact match first, then fuzzy match
        if name_lower not in valid_names:
            clean_key = re.sub(r"[^a-z0-9 ]", "", name_lower)
            if clean_key in clean_name_map:
                logger.info(f"Fuzzy matched '{name}' → '{clean_name_map[clean_key]}'")
                name_lower = clean_name_map[clean_key]
                # Find the original-case name from the DataFrame
                match = candidates_df[candidates_df["restaurant_name"].str.lower().str.strip() == name_lower]
                if not match.empty:
                    name = match.iloc[0]["restaurant_name"]
            else:
                logger.warning(f"Hallucinated restaurant '{name}' — skipping.")
                continue

        seen_names.add(name_lower)

        # Parse cuisines — handle both list and comma-separated string
        cuisines = item.get("cuisines", [])
        if isinstance(cuisines, str):
            cuisines = [c.strip() for c in cuisines.split(",") if c.strip()]

        # Parse rating safely
        try:
            rating = float(item.get("rating", 0))
        except (TypeError, ValueError):
            rating = 0.0

        # Parse cost safely
        try:
            cost = int(float(item.get("cost_for_two", 0)))
        except (TypeError, ValueError):
            cost = 0

        recommendations.append(Recommendation(
            rank=item.get("rank", len(recommendations) + 1),
            restaurant_name=name,
            cuisines=cuisines,
            rating=rating,
            cost_for_two=cost,
            location=item.get("location", ""),
            explanation=item.get("explanation", "No explanation provided."),
            trade_offs=item.get("trade_offs"),
        ))

    # Re-number ranks sequentially
    for i, rec in enumerate(recommendations):
        rec.rank = i + 1

    if not recommendations:
        raise LLMParseError(
            raw_response,
            f"JSON parsed but 0 recommendations matched candidates. "
            f"Valid names: {list(valid_names)[:5]}..."
        )

    logger.info(f"Parsed {len(recommendations)} valid recommendations.")
    return recommendations


def _extract_json(text: str) -> dict | list:
    """
    Extract JSON from LLM response.

    Handles common patterns from thinking models:
      - Clean JSON
      - JSON in ```json ... ``` fences
      - Preamble text before JSON
      - Multiple JSON blocks (takes the largest)
    """
    if not text or not text.strip():
        raise LLMParseError(text or "", "Empty response from LLM.")

    # Step 1: Try to find JSON inside markdown code fences
    fence_matches = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_matches:
        # Try each fence block, prefer the longest one (most likely the full response)
        for match in sorted(fence_matches, key=len, reverse=True):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Step 2: Try direct parse of the entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Step 3: Find the outermost JSON object or array
    # Use a bracket-counting approach for robustness
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        result = _find_balanced_json(text, start_char, end_char)
        if result is not None:
            return result

    # Step 4: Last resort — try to find any {...} or [...] substring
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    # Step 5: Truncation recovery — try to repair truncated JSON
    # Common with thinking models that hit max_output_tokens
    repaired = _try_repair_truncated_json(text)
    if repaired is not None:
        logger.warning("Recovered partial JSON from truncated response.")
        return repaired

    # Log what we received for debugging
    logger.error(f"Failed to extract JSON from response:\n{text[:1000]}")
    raise LLMParseError(text, "Could not extract valid JSON from response.")


def _find_balanced_json(text: str, open_char: str, close_char: str):
    """Find the first balanced JSON structure in the text."""
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == '\\' and in_string:
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None

    return None


def _try_repair_truncated_json(text: str):
    """
    Attempt to recover recommendations from a truncated JSON response.

    When the model hits max_output_tokens, the response is cut mid-JSON like:
        {"recommendations": [{"rank": 1, ...}, {"rank": 2, "restaurant_name": "Foo
    
    This function extracts all *complete* recommendation objects that were
    fully written before truncation.
    """
    # Find the start of the recommendations array
    arr_match = re.search(r'"recommendations"\s*:\s*\[', text)
    if not arr_match:
        # Maybe it's a bare array
        arr_start = text.find('[')
        if arr_start == -1:
            return None
    else:
        arr_start = arr_match.end() - 1  # position of the '['

    # Extract individual complete objects from the array
    # Find all balanced {...} blocks after the array start
    complete_items = []
    pos = arr_start + 1

    while pos < len(text):
        # Skip whitespace and commas
        while pos < len(text) and text[pos] in ' \t\n\r,':
            pos += 1

        if pos >= len(text) or text[pos] != '{':
            break

        # Try to find a balanced object starting here
        depth = 0
        in_string = False
        escape_next = False
        obj_start = pos

        for i in range(pos, len(text)):
            c = text[i]

            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    obj_text = text[obj_start:i + 1]
                    try:
                        obj = json.loads(obj_text)
                        if isinstance(obj, dict) and "restaurant_name" in obj:
                            complete_items.append(obj)
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            # Didn't find a closing brace — truncated mid-object
            break

        if depth != 0:
            break

    if complete_items:
        logger.info(
            f"Truncation recovery: salvaged {len(complete_items)} "
            f"complete recommendations from truncated response."
        )
        return {"recommendations": complete_items}

    return None
