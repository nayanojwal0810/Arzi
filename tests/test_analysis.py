"""Unit tests for Arzi SQL business analysis and metrics computation."""

from __future__ import annotations

import decimal
import json
import os
from datetime import date

import pytest

from arzi.analysis import (
    calculate_share_pct,
    check_partial_month,
    serialize_json_value,
    validate_analysis_metrics,
    EXPECTED_RECORD_COUNT,
)
from arzi.database import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_USER, TARGET_DB, get_connection_config


def test_percentage_calculation_valid_denominator() -> None:
    assert calculate_share_pct(25, 100) == 25.0
    assert calculate_share_pct(1, 3) == 33.33
    assert calculate_share_pct(55558, 117214) == 47.4


def test_percentage_calculation_zero_denominator() -> None:
    assert calculate_share_pct(10, 0) is None
    assert calculate_share_pct(0, 0) is None
    assert calculate_share_pct(5, -1) is None


def test_partial_month_detection() -> None:
    # Max date is 2020-01-09 (dataset end); Jan 2020 should be marked partial
    assert check_partial_month(date(2020, 1, 1), date(2020, 1, 9)) is True
    assert check_partial_month("2020-01-01", "2020-01-09") is True

    # Complete month: December 2019 when max date is 2020-01-09
    assert check_partial_month(date(2019, 12, 1), date(2020, 1, 9)) is False

    # Complete month: January 2020 if max date was 2020-01-31
    assert check_partial_month(date(2020, 1, 1), date(2020, 1, 31)) is False


def test_top_categories_ordering() -> None:
    synthetic_products = [
        {"product": "Mortgage", "complaint_count": 9449},
        {"product": "Credit reporting", "complaint_count": 55558},
        {"product": "Debt collection", "complaint_count": 22287},
    ]
    sorted_products = sorted(
        synthetic_products, key=lambda x: x["complaint_count"], reverse=True
    )
    assert sorted_products[0]["product"] == "Credit reporting"
    assert sorted_products[1]["product"] == "Debt collection"
    assert sorted_products[2]["product"] == "Mortgage"


def test_json_serialization_dates_and_decimals() -> None:
    test_dict = {
        "record_date": date(2019, 5, 20),
        "rate": decimal.Decimal("99.21"),
        "count": 100,
    }
    dumped = json.dumps(test_dict, default=serialize_json_value)
    loaded = json.loads(dumped)

    assert loaded["record_date"] == "2019-05-20"
    assert loaded["rate"] == 99.21
    assert loaded["count"] == 100


def test_json_serialization_unsupported_type() -> None:
    class CustomObject:
        pass

    with pytest.raises(TypeError, match="is not JSON serializable"):
        serialize_json_value(CustomObject())


def test_database_selection_unrelated_pgdatabase(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set an unrelated PGDATABASE value in the environment
    monkeypatch.setenv("PGDATABASE", "unrelated_external_database")

    # When explicit TARGET_DB is requested, config must use TARGET_DB
    config_explicit = get_connection_config(dbname=TARGET_DB)
    assert config_explicit["dbname"] == TARGET_DB

    # Default host and port should be standard defaults unless overridden
    assert config_explicit["host"] == DEFAULT_HOST
    assert config_explicit["port"] == DEFAULT_PORT
    assert config_explicit["user"] == DEFAULT_USER


def test_validate_analysis_metrics_synthetic_success() -> None:
    products = [{"complaint_count": EXPECTED_RECORD_COUNT, "complaint_share_pct": 100.0}]
    monthly = [
        {
            "complaint_count": EXPECTED_RECORD_COUNT,
            "is_partial_last_month": True,
            "month_over_month_change_pct": None,
        }
    ]
    issues = [{"complaint_count": EXPECTED_RECORD_COUNT}]
    responses = [{"complaint_count": EXPECTED_RECORD_COUNT}]
    channels = [{"complaint_count": EXPECTED_RECORD_COUNT}]

    # Should not raise any exception
    validate_analysis_metrics(products, monthly, issues, responses, channels)


def test_validate_analysis_metrics_failure_on_mismatch() -> None:
    products = [{"complaint_count": 100, "complaint_share_pct": 100.0}]
    monthly = [
        {
            "complaint_count": 100,
            "is_partial_last_month": False,
            "month_over_month_change_pct": 10.0,
        }
    ]
    issues = [{"complaint_count": 100}]
    responses = [{"complaint_count": 100}]
    channels = [{"complaint_count": 100}]

    with pytest.raises(ValueError, match="Analysis metrics validation failed"):
        validate_analysis_metrics(products, monthly, issues, responses, channels)
