"""Tests for arzi.data_audit module using synthetic data."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from arzi.data_audit import (
    audit_complaint_ids,
    audit_date_range,
    audit_dataset,
    audit_narratives,
    calculate_missing_values,
    calculate_sha256,
    check_file_exists,
    count_distinct_values,
    count_duplicate_rows,
    get_column_names,
    get_shape,
    identify_required_fields,
    verify_gzip_integrity,
)


def test_sha256_calculation(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.txt"
    content = b"financial complaint intelligence test content"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert calculate_sha256(test_file) == expected_hash


def test_missing_value_handling() -> None:
    df = pd.DataFrame({
        "col_a": [1, None, 3, None],
        "col_b": ["x", "y", "z", "w"],
        "col_c": [None, None, None, None],
    })
    missing = calculate_missing_values(df)
    assert missing["col_a"] == 2
    assert missing["col_b"] == 0
    assert missing["col_c"] == 4


def test_duplicate_complaint_id_detection() -> None:
    df = pd.DataFrame({
        "complaint_id": [101, 102, 103, 102, None, 104, 101],
    })
    audit = audit_complaint_ids(df)
    assert audit is not None
    assert audit["missing"] == 1
    assert audit["duplicates"] == 2


def test_complaint_id_missing_column() -> None:
    df = pd.DataFrame({"other_id": [1, 2, 3]})
    assert audit_complaint_ids(df) is None


def test_narrative_handling() -> None:
    df = pd.DataFrame({
        "consumer_complaint_narrative": [
            "Valid narrative text one.",
            "",                     # empty string
            "   ",                  # whitespace only
            None,                   # null
            "NA",                   # placeholder
            "Valid narrative text two.",
            "Valid narrative text one.",  # duplicate
        ]
    })
    audit = audit_narratives(df)
    assert audit is not None
    assert audit["usable_count"] == 3
    assert audit["unusable_count"] == 4
    assert audit["duplicate_count"] == 1


def test_basic_date_parsing() -> None:
    df = pd.DataFrame({
        "date_received": [
            "2019-03-15",
            "2019-01-01",
            "invalid-date",
            "2019-12-31",
            None,
        ]
    })
    min_date, max_date = audit_date_range(df)
    assert min_date == "2019-01-01"
    assert max_date == "2019-12-31"


def test_handling_missing_input_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.csv.gz"
    assert not check_file_exists(non_existent)

    with pytest.raises(FileNotFoundError):
        calculate_sha256(non_existent)

    audit_result = audit_dataset(non_existent)
    assert not audit_result["file_exists"]
    assert not audit_result["verification_passed"]
    assert "Dataset file does not exist." in audit_result["failure_reasons"]


def test_synthetic_gzip_dataset_audit(tmp_path: Path) -> None:
    csv_gz_path = tmp_path / "test_complaints.csv.gz"
    sample_df = pd.DataFrame({
        "complaint_id": [1, 2, 3],
        "date_received": ["2019-05-01", "2019-05-02", "2019-05-03"],
        "product": ["Credit card", "Mortgage", "Credit card"],
        "issue": ["Billing", "Loan servicing", "Billing"],
        "company": ["Bank A", "Bank B", "Bank A"],
        "consumer_complaint_narrative": ["Issue with billing", "Servicing issue", "Issue with billing"],
        "company_response_to_consumer": ["Closed with explanation", "Closed with explanation", "Closed with monetary relief"],
    })

    with gzip.open(csv_gz_path, "wt", encoding="utf-8") as gz_file:
        sample_df.to_csv(gz_file, index=False)

    assert verify_gzip_integrity(csv_gz_path)
    file_sha = calculate_sha256(csv_gz_path)

    audit = audit_dataset(csv_gz_path, expected_sha256=file_sha)
    assert audit["verification_passed"]
    assert audit["total_rows"] == 3
    assert audit["total_columns"] == 7
    assert audit["narrative_audit"]["usable_count"] == 3
    assert audit["narrative_audit"]["duplicate_count"] == 1
    assert audit["distinct_counts"]["product"] == 2
    assert audit["distinct_counts"]["company"] == 2


def test_shape_and_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert get_shape(df) == (2, 2)
    assert get_column_names(df) == ["a", "b"]
    assert count_duplicate_rows(df) == 0


def test_identify_required_fields() -> None:
    df = pd.DataFrame({
        "complaint_id": [1],
        "date_received": ["2019-01-01"],
        "product": ["Debt collection"],
        "issue": ["Written notification"],
        "company": ["Acme Inc"],
        "consumer_complaint_narrative": ["Complaint narrative"],
        "company_response_to_consumer": ["Closed with explanation"],
    })
    identified = identify_required_fields(df)
    assert identified["complaint_id"] == "complaint_id"
    assert identified["date_received"] == "date_received"
    assert identified["product"] == "product"
    assert identified["issue"] == "issue"
    assert identified["company"] == "company"
    assert identified["consumer_complaint_narrative"] == "consumer_complaint_narrative"
    assert identified["company_response"] == "company_response_to_consumer"
