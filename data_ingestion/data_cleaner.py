"""
Phase 1 — Data Cleaner

Clean, normalize, and validate the raw Zomato dataset.

Functions:
    clean(df)           -> pd.DataFrame
    normalize_text(col) -> pd.Series
    cast_types(df)      -> pd.DataFrame
"""

from __future__ import annotations

import logging
import unicodedata

import pandas as pd

from config import settings, CITY_ALIASES
from models import SchemaError, DataQualityError

logger = logging.getLogger(__name__)

# Columns we need from the raw dataset (mapped to our clean names)
# Keys = possible raw column names, Values = our standard name
_COLUMN_MAP: dict[str, str] = {
    "name":                  "restaurant_name",
    "restaurant name":       "restaurant_name",
    "city":                  "city",
    "location":              "city",
    "cuisines":              "cuisines",
    "cuisine":               "cuisines",
    "average cost for two":  "average_cost_for_two",
    "approx_cost(for two people)": "average_cost_for_two",
    "cost":                  "average_cost_for_two",
    "aggregate rating":      "aggregate_rating",
    "rate":                  "aggregate_rating",
    "rating":                "aggregate_rating",
    "votes":                 "votes",
    "has online delivery":   "has_online_delivery",
    "online_order":          "has_online_delivery",
}

# Null threshold — raise DataQualityError if exceeded
_NULL_THRESHOLD = 0.30  # 30%


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: rename columns, select fields, handle nulls,
    normalize text, cast types, and deduplicate.

    Args:
        df: Raw DataFrame from the data loader.

    Returns:
        Clean, schema-validated DataFrame.

    Raises:
        SchemaError: If critical columns are missing.
        DataQualityError: If null percentage exceeds threshold.
    """
    logger.info(f"Cleaning dataset: {len(df)} rows, {len(df.columns)} columns.")

    # Step 1: Normalize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # Step 2: Rename columns to our standard names
    rename_map = {}
    for raw_name in df.columns:
        if raw_name in _COLUMN_MAP:
            rename_map[raw_name] = _COLUMN_MAP[raw_name]
    df = df.rename(columns=rename_map)

    # Step 3: Validate required columns exist
    required = set(settings.expected_schema.keys())
    available = set(df.columns)
    missing = required - available
    if missing:
        # Try to find partial matches before raising
        logger.error(f"Missing columns: {missing}. Available: {sorted(available)}")
        raise SchemaError(missing_columns=sorted(missing))

    # Step 4: Select only the columns we need
    df = df[list(required)].copy()

    # Step 5: Check null percentages
    _check_null_quality(df)

    # Step 6: Clean each column
    df["restaurant_name"] = normalize_text(df["restaurant_name"])
    df["city"] = _normalize_city(df["city"])
    df["cuisines"] = _parse_cuisines(df["cuisines"])
    df = cast_types(df)

    # Step 7: Drop rows with nulls in critical columns
    critical = ["restaurant_name", "city", "aggregate_rating"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null values in critical columns.")

    # Step 8: Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["restaurant_name", "city"], keep="first")
    deduped = before - len(df)
    if deduped:
        logger.info(f"Removed {deduped} duplicate entries.")

    # Step 9: Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Cleaning complete: {len(df)} restaurants retained.")
    return df


def normalize_text(col: pd.Series) -> pd.Series:
    """
    Normalize a text column: strip whitespace, NFC unicode normalization,
    and title-case.
    """
    return col.astype(str).apply(
        lambda x: unicodedata.normalize("NFC", x.strip()).title()
        if pd.notna(x) and x.strip() else x
    )


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to their expected types with safe conversions.
    Non-convertible values become NaN and are filled with defaults.
    """
    # Average cost → integer
    df["average_cost_for_two"] = (
        pd.to_numeric(df["average_cost_for_two"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Rating → float, clamped to [0.0, 5.0]
    # Raw values may be strings like "4.1/5" — strip the "/5" suffix first
    df["aggregate_rating"] = df["aggregate_rating"].apply(_parse_rating)
    df["aggregate_rating"] = df["aggregate_rating"].clip(0.0, 5.0)

    # Votes → integer
    df["votes"] = (
        pd.to_numeric(df["votes"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Online delivery → boolean
    df["has_online_delivery"] = df["has_online_delivery"].apply(_parse_bool)

    return df


# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------

def _normalize_city(col: pd.Series) -> pd.Series:
    """Lowercase, strip, and apply city alias mapping."""
    def _clean_city(val):
        if pd.isna(val) or not str(val).strip():
            return "unknown"
        city = str(val).strip().lower()
        city = unicodedata.normalize("NFC", city)
        return CITY_ALIASES.get(city, city)

    return col.apply(_clean_city)


def _parse_cuisines(col: pd.Series) -> pd.Series:
    """
    Parse cuisine strings into lists.
    'North Indian, Chinese, BBQ' → ['north indian', 'chinese', 'bbq']
    """
    def _split(val):
        if pd.isna(val) or not str(val).strip():
            return ["unknown"]
        items = [c.strip().lower() for c in str(val).split(",") if c.strip()]
        return items if items else ["unknown"]

    return col.apply(_split)


def _parse_bool(val) -> bool:
    """Convert various truthy/falsy values to bool."""
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    return str(val).strip().lower() in {"yes", "1", "true"}


def _parse_rating(val) -> float:
    """
    Parse rating values like '4.1/5', '4.1', 'NEW', '-', etc.
    Returns a float between 0.0 and 5.0.
    """
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    # Handle 'NEW', '-', empty strings
    if not val_str or val_str.lower() in {"new", "-", "nan", "none", ""}:
        return 0.0
    # Handle '4.1/5' format
    if "/" in val_str:
        val_str = val_str.split("/")[0].strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def _check_null_quality(df: pd.DataFrame) -> None:
    """Raise DataQualityError if any column exceeds the null threshold."""
    for col in df.columns:
        null_pct = df[col].isna().mean()
        if null_pct > _NULL_THRESHOLD:
            logger.error(
                f"Column '{col}' has {null_pct:.1%} null values "
                f"(threshold: {_NULL_THRESHOLD:.0%})."
            )
            raise DataQualityError(column=col, null_percentage=null_pct * 100)
