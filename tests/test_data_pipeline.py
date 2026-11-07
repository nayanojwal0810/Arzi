"""Unit tests for Arzi data preparation pipeline using synthetic records."""

from __future__ import annotations

import datetime
import pytest

from arzi.data_pipeline import (
    clean_narrative,
    clean_text_field,
    parse_boolean_yes_no,
    parse_complaint_id,
    parse_date,
    prepare_record,
    validate_csv_header,
    EXPECTED_COLUMNS,
)


def test_valid_complaint_id_conversion() -> None:
    assert parse_complaint_id("3384392") == 3384392
    assert parse_complaint_id(12345) == 12345
    assert parse_complaint_id("  9999999  ") == 9999999


def test_invalid_complaint_id_handling() -> None:
    with pytest.raises(ValueError, match="Missing complaint_id"):
        parse_complaint_id(None)
    with pytest.raises(ValueError, match="Missing complaint_id"):
        parse_complaint_id("   ")
    with pytest.raises(ValueError, match="cannot convert to integer"):
        parse_complaint_id("invalid-id-xyz")


def test_correct_date_parsing() -> None:
    parsed = parse_date("2019-03-15", required=True)
    assert parsed == datetime.date(2019, 3, 15)

    optional_parsed = parse_date(None, required=False)
    assert optional_parsed is None

    empty_optional = parse_date("   ", required=False)
    assert empty_optional is None


def test_invalid_date_handling() -> None:
    with pytest.raises(ValueError, match="Required date is missing"):
        parse_date(None, required=True)
    with pytest.raises(ValueError, match="Invalid date value"):
        parse_date("2019/03/15", required=True)
    with pytest.raises(ValueError, match="Invalid date value"):
        parse_date("not-a-date", required=True)


def test_missing_optional_field_handling() -> None:
    assert clean_text_field(None, required=False) is None
    assert clean_text_field("   ", required=False) is None
    assert clean_text_field("  Valid text  ", required=False) == "Valid text"


def test_narrative_preservation() -> None:
    raw_text = "I received a billing error on my statement! Please help."
    text, has_narrative = clean_narrative(raw_text)
    assert text == raw_text
    assert has_narrative is True

    # Empty, whitespace, or placeholders
    assert clean_narrative(None) == (None, False)
    assert clean_narrative("   ") == (None, False)
    assert clean_narrative("NA") == (None, False)
    assert clean_narrative("N/A") == (None, False)


def test_correct_yes_no_conversion_timely_response() -> None:
    assert parse_boolean_yes_no("Yes") is True
    assert parse_boolean_yes_no("yes") is True
    assert parse_boolean_yes_no("No") is False
    assert parse_boolean_yes_no("no") is False
    assert parse_boolean_yes_no(None) is None
    assert parse_boolean_yes_no("   ") is None
    assert parse_boolean_yes_no("Unknown") is None


def test_required_field_validation() -> None:
    with pytest.raises(ValueError, match="Required field 'product' is empty"):
        clean_text_field("", required=True, field_name="product")
    with pytest.raises(ValueError, match="Required field 'company' is None"):
        clean_text_field(None, required=True, field_name="company")


def test_prepare_record_end_to_end() -> None:
    synthetic_row = {
        "complaint_id": " 5551234 ",
        "date_received": "2019-08-20",
        "date_sent_to_company": "2019-08-21",
        "product": " Credit card ",
        "sub_product": " General-purpose credit card ",
        "issue": " Billing disputes ",
        "sub_issue": " Incorrect fee charged ",
        "consumer_complaint_narrative": "Disputed fee of $25 on statement.",
        "company_public_response": " Company has responded to the consumer ",
        "company": " Acme Bank ",
        "state": " CA ",
        "zip_code": " 02138 ",
        "tags": " Servicemember ",
        "consumer_consent_provided": " Consent provided ",
        "submitted_via": " Web ",
        "company_response_to_consumer": " Closed with explanation ",
        "timely_response": "Yes",
        "consumer_disputed": " ",
    }

    prepared = prepare_record(synthetic_row)
    assert prepared[0] == 5551234  # complaint_id int
    assert prepared[1] == datetime.date(2019, 8, 20)  # date_received
    assert prepared[2] == datetime.date(2019, 8, 21)  # date_sent_to_company
    assert prepared[3] == "Credit card"  # product
    assert prepared[5] == "Billing disputes"  # issue
    assert prepared[7] == "Disputed fee of $25 on statement."  # narrative preserved
    assert prepared[9] == "Acme Bank"  # company
    assert prepared[11] == "02138"  # zip_code preserves leading zero
    assert prepared[16] is True  # timely_response
    assert prepared[17] is None  # consumer_disputed empty -> None
    assert prepared[18] is True  # has_narrative


def test_validate_csv_header() -> None:
    validate_csv_header(EXPECTED_COLUMNS)

    with pytest.raises(ValueError, match="Duplicate columns"):
        validate_csv_header(EXPECTED_COLUMNS + ["product"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_csv_header(EXPECTED_COLUMNS[:-1])

    with pytest.raises(ValueError, match="Unexpected extra columns"):
        validate_csv_header(EXPECTED_COLUMNS + ["extra_column"])
