"""Data preparation and ingestion pipeline for Arzi."""

from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from arzi.database import (
    apply_schema,
    ensure_database_exists,
    get_connection,
    get_server_version,
    load_prepared_records_copy,
    load_raw_records_copy,
    run_validation_checks,
    truncate_target_tables,
)

EXPECTED_SHA256 = "f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b"

EXPECTED_COLUMNS: list[str] = [
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "consumer_consent_provided",
    "submitted_via",
    "date_sent_to_company",
    "company_response_to_consumer",
    "timely_response",
    "consumer_disputed",
    "complaint_id",
]

PREPARED_COLUMNS: list[str] = [
    "complaint_id",
    "date_received",
    "date_sent_to_company",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "consumer_consent_provided",
    "submitted_via",
    "company_response_to_consumer",
    "timely_response",
    "consumer_disputed",
    "has_narrative",
]

PLACEHOLDER_NARRATIVES = frozenset({"na", "n/a", "null", "none", "nan", "nil", ""})


def calculate_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 checksum of a file reading in chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_source_file(
    file_path: Path | str, expected_sha256: str = EXPECTED_SHA256
) -> str:
    """Verify source file existence and cryptographic checksum."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found: {path}")

    actual_sha = calculate_sha256(path)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha}"
        )
    return actual_sha


def validate_csv_header(actual_fields: list[str]) -> None:
    """Validate that the source header contains exactly the expected columns."""
    actual_set = set(actual_fields)
    expected_set = set(EXPECTED_COLUMNS)

    if len(actual_fields) != len(actual_set):
        duplicates = [col for col in actual_set if actual_fields.count(col) > 1]
        raise ValueError(f"Duplicate columns detected in CSV header: {duplicates}")

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:
        raise ValueError(f"Missing required columns in CSV header: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra columns in CSV header: {sorted(extra)}")


def parse_complaint_id(val: Any) -> int:
    """Safely convert a complaint identifier to an integer."""
    if val is None:
        raise ValueError("Missing complaint_id: value is None")
    s = str(val).strip()
    if not s:
        raise ValueError("Missing complaint_id: empty string")
    try:
        return int(s)
    except ValueError as exc:
        raise ValueError(f"Invalid complaint_id '{val}': cannot convert to integer") from exc


def parse_date(val: Any, required: bool = True) -> datetime.date | None:
    """Parse an ISO date string (%Y-%m-%d) into a date object."""
    if val is None:
        if required:
            raise ValueError("Required date is missing (None)")
        return None

    s = str(val).strip()
    if not s:
        if required:
            raise ValueError("Required date is missing (empty string)")
        return None

    try:
        return datetime.date.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"Invalid date value '{val}', expected YYYY-MM-DD") from exc


def clean_text_field(
    val: Any, required: bool = False, field_name: str = ""
) -> str | None:
    """Clean categorical text by stripping whitespace, returning None for empty strings."""
    if val is None:
        if required:
            raise ValueError(f"Required field '{field_name}' is None")
        return None

    s = str(val).strip()
    if not s:
        if required:
            raise ValueError(f"Required field '{field_name}' is empty")
        return None
    return s


def parse_boolean_yes_no(val: Any) -> bool | None:
    """Convert confirmed Yes/No values to True/False, returning None for others."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s == "yes":
        return True
    if s == "no":
        return False
    return None


def clean_narrative(val: Any) -> tuple[str | None, bool]:
    """Preserve raw narrative text while computing the has_narrative indicator."""
    if val is None:
        return None, False

    s = str(val)
    stripped = s.strip()
    if not stripped or stripped.lower() in PLACEHOLDER_NARRATIVES:
        return None, False

    return s, True


def prepare_record(row: dict[str, str]) -> tuple[Any, ...]:
    """Clean and standardize a single complaint dictionary into a prepared tuple."""
    complaint_id = parse_complaint_id(row.get("complaint_id"))
    date_received = parse_date(row.get("date_received"), required=True)
    date_sent_to_company = parse_date(row.get("date_sent_to_company"), required=False)

    product = clean_text_field(row.get("product"), required=True, field_name="product")
    sub_product = clean_text_field(row.get("sub_product"), required=False)
    issue = clean_text_field(row.get("issue"), required=True, field_name="issue")
    sub_issue = clean_text_field(row.get("sub_issue"), required=False)

    narrative, has_narrative = clean_narrative(row.get("consumer_complaint_narrative"))
    company_public_response = clean_text_field(row.get("company_public_response"), required=False)
    company = clean_text_field(row.get("company"), required=True, field_name="company")
    state = clean_text_field(row.get("state"), required=False)
    zip_code = clean_text_field(row.get("zip_code"), required=False)
    tags = clean_text_field(row.get("tags"), required=False)
    consumer_consent_provided = clean_text_field(row.get("consumer_consent_provided"), required=False)
    submitted_via = clean_text_field(row.get("submitted_via"), required=False)
    company_response_to_consumer = clean_text_field(row.get("company_response_to_consumer"), required=False)
    timely_response = parse_boolean_yes_no(row.get("timely_response"))
    consumer_disputed = clean_text_field(row.get("consumer_disputed"), required=False)

    return (
        complaint_id,
        date_received,
        date_sent_to_company,
        product,
        sub_product,
        issue,
        sub_issue,
        narrative,
        company_public_response,
        company,
        state,
        zip_code,
        tags,
        consumer_consent_provided,
        submitted_via,
        company_response_to_consumer,
        timely_response,
        consumer_disputed,
        has_narrative,
    )


def iter_raw_records(file_path: Path | str) -> Iterator[tuple[Any, ...]]:
    """Stream raw records as tuples corresponding to EXPECTED_COLUMNS."""
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")
        validate_csv_header(reader.fieldnames)
        for row in reader:
            yield tuple(row.get(col) for col in EXPECTED_COLUMNS)


def iter_prepared_records(file_path: Path | str) -> Iterator[tuple[Any, ...]]:
    """Stream prepared records as validated tuples corresponding to PREPARED_COLUMNS."""
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")
        validate_csv_header(reader.fieldnames)
        for row in reader:
            yield prepare_record(row)


def validate_pipeline_metrics(metrics: dict[str, Any]) -> None:
    """Verify that loaded data conforms to the required acceptance criteria."""
    expected_count = 117214
    expected_repeated_narratives = 10854
    expected_min_date = "2019-01-01"
    expected_max_date = "2020-01-09"

    errors: list[str] = []

    if metrics.get("raw_count") != expected_count:
        errors.append(
            f"Raw table count mismatch: expected {expected_count:,}, got {metrics.get('raw_count')}"
        )

    if metrics.get("prepared_count") != expected_count:
        errors.append(
            f"Prepared table count mismatch: expected {expected_count:,}, got {metrics.get('prepared_count')}"
        )

    if metrics.get("distinct_ids") != expected_count:
        errors.append(
            f"Distinct complaint IDs mismatch: expected {expected_count:,}, got {metrics.get('distinct_ids')}"
        )

    if metrics.get("missing_ids") != 0:
        errors.append(f"Missing complaint IDs found: {metrics.get('missing_ids')}")

    if metrics.get("duplicate_ids") != 0:
        errors.append(f"Duplicate complaint IDs found: {metrics.get('duplicate_ids')}")

    if metrics.get("min_date") != expected_min_date or metrics.get("max_date") != expected_max_date:
        errors.append(
            f"Date range mismatch: expected {expected_min_date} to {expected_max_date}, "
            f"got {metrics.get('min_date')} to {metrics.get('max_date')}"
        )

    if metrics.get("repeated_narratives") != expected_repeated_narratives:
        errors.append(
            f"Repeated narrative count mismatch: expected {expected_repeated_narratives:,}, "
            f"got {metrics.get('repeated_narratives')}"
        )

    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(f"  - {err}" for err in errors))


def run_pipeline(
    dataset_path: Path | str, schema_path: Path | str
) -> dict[str, Any]:
    """Execute complete end-to-end data preparation and loading workflow."""
    ds_path = Path(dataset_path)
    sc_path = Path(schema_path)

    # 1. Verify source file checksum
    checksum = verify_source_file(ds_path)

    # 2. Ensure target database exists
    ensure_database_exists()

    # 3. Connect to target database specifically and apply schema
    with get_connection(dbname="arzi") as conn:
        server_version = get_server_version(conn)
        apply_schema(conn, sc_path)
        conn.commit()

        # 4. In a controlled transaction, truncate project tables, load, and validate
        try:
            truncate_target_tables(conn)

            raw_loaded = load_raw_records_copy(
                conn, EXPECTED_COLUMNS, iter_raw_records(ds_path)
            )
            prepared_loaded = load_prepared_records_copy(
                conn, PREPARED_COLUMNS, iter_prepared_records(ds_path)
            )

            metrics = run_validation_checks(conn)
            metrics["server_version"] = server_version
            metrics["checksum"] = checksum
            metrics["raw_loaded"] = raw_loaded
            metrics["prepared_loaded"] = prepared_loaded
            metrics["db_host"] = conn.info.host or "localhost"
            metrics["db_port"] = conn.info.port or 5432
            metrics["db_name"] = conn.info.dbname or "arzi"

            # 5. Validate metrics against acceptance conditions (raises ValueError on failure)
            validate_pipeline_metrics(metrics)

            # 6. Commit transaction only after all validation checks succeed
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return metrics
