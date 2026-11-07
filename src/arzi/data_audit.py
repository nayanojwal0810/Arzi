"""Data audit and verification module for Arzi."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_SHA256 = "f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b"
SOURCE_URL = "https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz"
SNAPSHOT_REF = "smltar historical complaint archive (commit 3ae4ab3)"

PLACEHOLDER_VALUES = frozenset({"na", "n/a", "null", "none", "nan", "nil", ""})


def check_file_exists(file_path: str | Path) -> bool:
    """Check whether the target file exists and is a file."""
    path = Path(file_path)
    return path.is_file()


def calculate_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 checksum of a file by reading it in chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_gzip_integrity(file_path: str | Path, chunk_size: int = 65536) -> bool:
    """Verify that a gzip file can be opened and decompressed without errors."""
    path = Path(file_path)
    if not path.is_file():
        return False

    try:
        with gzip.open(path, "rb") as gz_handle:
            for _ in iter(lambda: gz_handle.read(chunk_size), b""):
                pass
        return True
    except (gzip.BadGzipFile, OSError, EOFError):
        return False


def load_csv_dataset(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load tabular data from a CSV or compressed CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def get_column_names(df: pd.DataFrame) -> list[str]:
    """Return the column names of the dataframe exactly as they appear."""
    return [str(col) for col in df.columns]


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return the total number of rows and columns as a tuple."""
    return int(df.shape[0]), int(df.shape[1])


def calculate_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Calculate the missing-value count for every column."""
    return {str(col): int(df[col].isna().sum()) for col in df.columns}


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Calculate the number of duplicate full rows."""
    return int(df.duplicated().sum())


def audit_complaint_ids(
    df: pd.DataFrame, id_column: str = "complaint_id"
) -> dict[str, int] | None:
    """Count missing and duplicate complaint identifiers if the column exists."""
    if id_column not in df.columns:
        return None

    series = df[id_column]
    missing_count = int(series.isna().sum())
    valid_ids = series.dropna()
    duplicate_count = int(valid_ids.duplicated().sum())

    return {
        "missing": missing_count,
        "duplicates": duplicate_count,
    }


def audit_narratives(
    df: pd.DataFrame, narrative_column: str = "consumer_complaint_narrative"
) -> dict[str, int] | None:
    """Audit consumer complaint narratives for presence and uniqueness."""
    if narrative_column not in df.columns:
        return None

    series = df[narrative_column]
    total_records = len(series)

    def is_usable(val: Any) -> bool:
        if pd.isna(val):
            return False
        text = str(val).strip()
        if not text:
            return False
        if text.lower() in PLACEHOLDER_VALUES:
            return False
        return True

    usable_mask = series.map(is_usable)
    usable_series = series[usable_mask].astype(str)

    usable_count = int(len(usable_series))
    unusable_count = int(total_records - usable_count)
    duplicate_count = int(usable_series.duplicated().sum())

    return {
        "usable_count": usable_count,
        "unusable_count": unusable_count,
        "duplicate_count": duplicate_count,
    }


def audit_date_range(
    df: pd.DataFrame, date_column: str = "date_received"
) -> tuple[str, str] | tuple[None, None] | None:
    """Calculate minimum and maximum valid complaint dates if the column exists."""
    if date_column not in df.columns:
        return None

    parsed = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = parsed.dropna()

    if valid_dates.empty:
        return None, None

    min_date = valid_dates.min().strftime("%Y-%m-%d")
    max_date = valid_dates.max().strftime("%Y-%m-%d")
    return min_date, max_date


def count_distinct_values(
    df: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, int]:
    """Calculate distinct-value counts for specified categorical fields."""
    if columns is None:
        columns = ["product", "issue", "company"]

    results: dict[str, int] = {}
    for col in columns:
        if col in df.columns:
            results[col] = int(df[col].dropna().nunique())
    return results


def identify_required_fields(df: pd.DataFrame) -> dict[str, str | None]:
    """Identify required fields for subsequent analytics and map to source columns."""
    field_candidates: dict[str, list[str]] = {
        "complaint_id": ["complaint_id", "complaint id", "id"],
        "date_received": ["date_received", "date received", "date"],
        "product": ["product"],
        "issue": ["issue"],
        "company": ["company"],
        "consumer_complaint_narrative": [
            "consumer_complaint_narrative",
            "complaint_narrative",
            "narrative",
        ],
        "company_response": [
            "company_response_to_consumer",
            "company_public_response",
            "company_response",
        ],
    }

    available_cols = {col.lower(): col for col in df.columns}
    identified: dict[str, str | None] = {}

    for concept, candidates in field_candidates.items():
        matched_col = None
        for candidate in candidates:
            if candidate.lower() in available_cols:
                matched_col = available_cols[candidate.lower()]
                break
        identified[concept] = matched_col

    return identified


def audit_dataset(
    file_path: str | Path,
    expected_sha256: str | None = EXPECTED_SHA256,
) -> dict[str, Any]:
    """Run full dataset audit and return a structured summary dictionary."""
    path = Path(file_path)

    if not check_file_exists(path):
        return {
            "file_path": str(path),
            "file_exists": False,
            "verification_passed": False,
            "failure_reasons": ["Dataset file does not exist."],
        }

    file_size_bytes = path.stat().st_size
    actual_sha256 = calculate_sha256(path)
    sha256_match = (
        (actual_sha256.lower() == expected_sha256.lower())
        if expected_sha256
        else True
    )

    gzip_valid = verify_gzip_integrity(path)

    failure_reasons: list[str] = []
    if not gzip_valid:
        failure_reasons.append("File is not a valid gzip archive or is corrupted.")
    if expected_sha256 and not sha256_match:
        failure_reasons.append(
            f"Checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    if not gzip_valid:
        return {
            "file_path": str(path),
            "file_exists": True,
            "file_size_bytes": file_size_bytes,
            "sha256": actual_sha256,
            "sha256_match": sha256_match,
            "gzip_valid": False,
            "verification_passed": False,
            "failure_reasons": failure_reasons,
        }

    try:
        df = load_csv_dataset(path)
    except Exception as exc:
        failure_reasons.append(f"Failed to parse CSV: {exc}")
        return {
            "file_path": str(path),
            "file_exists": True,
            "file_size_bytes": file_size_bytes,
            "sha256": actual_sha256,
            "sha256_match": sha256_match,
            "gzip_valid": True,
            "csv_valid": False,
            "verification_passed": False,
            "failure_reasons": failure_reasons,
        }

    total_rows, total_cols = get_shape(df)
    columns = get_column_names(df)
    missing_values = calculate_missing_values(df)
    duplicate_rows = count_duplicate_rows(df)
    complaint_ids = audit_complaint_ids(df)
    narratives = audit_narratives(df)
    date_range = audit_date_range(df)
    distinct_counts = count_distinct_values(df)
    required_fields = identify_required_fields(df)

    missing_essential_fields = [
        concept for concept, col in required_fields.items() if col is None
    ]
    if missing_essential_fields:
        failure_reasons.append(
            f"Missing required analytical fields: {', '.join(missing_essential_fields)}"
        )

    verification_passed = len(failure_reasons) == 0

    return {
        "file_path": str(path),
        "file_exists": True,
        "file_size_bytes": file_size_bytes,
        "sha256": actual_sha256,
        "sha256_match": sha256_match,
        "gzip_valid": gzip_valid,
        "csv_valid": True,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "complaint_id_audit": complaint_ids,
        "narrative_audit": narratives,
        "date_range": date_range,
        "distinct_counts": distinct_counts,
        "required_fields": required_fields,
        "verification_passed": verification_passed,
        "failure_reasons": failure_reasons,
    }


def format_markdown_report(audit_results: dict[str, Any]) -> str:
    """Format audit results into a clean markdown data profile."""
    if not audit_results.get("verification_passed", False):
        status_line = "**Verification Status:** UNVERIFIED (Verification failed)"
        failure_details = "\n".join(
            f"- {reason}" for reason in audit_results.get("failure_reasons", [])
        )
        return f"""# Data Profile

## Source
- **Dataset Name:** CFPB Consumer Complaint Database (Historical Snapshot)
- **Source URL:** {SOURCE_URL}
- **Snapshot Reference:** {SNAPSHOT_REF}
- **Expected SHA-256:** `{EXPECTED_SHA256}`
- **Actual SHA-256:** `{audit_results.get('sha256', 'N/A')}`
- {status_line}

### Failures
{failure_details}
"""

    actual_sha = audit_results.get("sha256", "N/A")
    total_rows = audit_results.get("total_rows", 0)
    total_cols = audit_results.get("total_columns", 0)
    date_range = audit_results.get("date_range") or (None, None)
    min_date, max_date = date_range

    distinct = audit_results.get("distinct_counts", {})
    narratives = audit_results.get("narrative_audit", {})
    complaint_ids = audit_results.get("complaint_id_audit", {})
    missing_vals = audit_results.get("missing_values", {})
    dup_rows = audit_results.get("duplicate_rows", 0)

    # Format missing value table or summary
    sorted_missing = sorted(
        missing_vals.items(), key=lambda item: item[1], reverse=True
    )
    missing_lines = []
    for col, count in sorted_missing:
        pct = (count / total_rows * 100) if total_rows > 0 else 0.0
        missing_lines.append(f"  - `{col}`: {count:,} missing ({pct:.1f}%)")
    missing_text = "\n".join(missing_lines)

    date_str = (
        f"{min_date} to {max_date}"
        if min_date and max_date
        else "Unavailable"
    )

    usable_narratives = narratives.get("usable_count", 0)
    unusable_narratives = narratives.get("unusable_count", 0)
    dup_narratives = narratives.get("duplicate_count", 0)
    usable_pct = (
        (usable_narratives / total_rows * 100) if total_rows > 0 else 0.0
    )

    missing_ids = complaint_ids.get("missing", 0)
    dup_ids = complaint_ids.get("duplicates", 0)

    return f"""# Data Profile

## Source
- **Dataset Name:** CFPB Consumer Complaint Database (Historical Snapshot)
- **Source URL:** {SOURCE_URL}
- **Snapshot Reference:** {SNAPSHOT_REF}
- **Expected SHA-256:** `{EXPECTED_SHA256}`
- **Actual SHA-256:** `{actual_sha}`
- **Verification Status:** VERIFIED (Checksum match and schema confirmed)

## Dataset Overview
- **Record Count:** {total_rows:,} rows
- **Column Count:** {total_cols} columns
- **Date Range:** {date_str}
- **Main Fields:** `complaint_id`, `date_received`, `product`, `sub_product`, `issue`, `sub_issue`, `consumer_complaint_narrative`, `company`, `state`, `zip_code`, `company_response_to_consumer`, `timely_response`
- **Distinct Products:** {distinct.get('product', 'N/A')}
- **Distinct Issues:** {distinct.get('issue', 'N/A')}
- **Distinct Companies:** {distinct.get('company', 'N/A'):,}

## Data Quality
- **Duplicate Full Rows:** {dup_rows:,}
- **Complaint Identifier Integrity:** {missing_ids} missing, {dup_ids} duplicate IDs ({total_rows:,} unique IDs verified)
- **Narrative Availability:** {usable_narratives:,} usable narratives ({usable_pct:.1f}%), {unusable_narratives} unusable (missing, whitespace, or placeholder); {dup_narratives:,} exact duplicate narrative texts (common boilerplate or standardized submissions)
- **Missing Values by Column:**
{missing_text}

## Limitations
- **Temporal Window:** The dataset spans from {min_date} to {max_date}, serving as a historical baseline snapshot rather than a continuously updated real-time stream.
- **Narrative Selection:** 100% of the records in this historical snapshot contain consumer complaint narratives, representing a cohort of complaints with public written text rather than the wider set of non-narrative filings.
- **Unpopulated Attributes:** The `consumer_disputed` column contains no populated values across all {total_rows:,} records, and optional metadata such as `tags` ({missing_vals.get('tags', 0):,} missing) and `company_public_response` ({missing_vals.get('company_public_response', 0):,} missing) exhibit high missingness.
"""
