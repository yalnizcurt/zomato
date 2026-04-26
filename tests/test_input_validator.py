import pytest

from models import BudgetLevel, InputValidationError
from user_input.input_validator import validate


def _options():
    return ["bangalore", "delhi"], ["italian", "chinese"]


def test_validate_handles_non_string_budget_with_validation_error():
    cities, cuisines = _options()

    with pytest.raises(InputValidationError) as exc:
        validate(
            {
                "location": "bangalore",
                "budget": 123,
                "cuisines": [],
                "min_rating": 4.0,
            },
            cities,
            cuisines,
        )

    assert exc.value.field_name == "budget"


def test_validate_accepts_string_cuisine_input_from_web_forms():
    cities, cuisines = _options()

    prefs = validate(
        {
            "location": "bangalore",
            "budget": "medium",
            "cuisines": "italian, chinese",
            "min_rating": 4.0,
        },
        cities,
        cuisines,
    )

    assert prefs.budget is BudgetLevel.MEDIUM
    assert prefs.cuisines == ["italian", "chinese"]


def test_validate_rejects_non_string_location_cleanly():
    cities, cuisines = _options()

    with pytest.raises(InputValidationError) as exc:
        validate(
            {
                "location": 999,
                "budget": "low",
                "cuisines": [],
                "min_rating": 3.0,
            },
            cities,
            cuisines,
        )

    assert exc.value.field_name == "location"
