#!/usr/bin/env python3
"""Dataset verification script for Arzi."""

from __future__ import annotations

import sys
from pathlib import Path

from arzi.data_audit import (
    EXPECTED_SHA256,
    audit_dataset,
    format_markdown_report,
)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "raw" / "complaints.csv.gz"
    report_path = project_root / "reports" / "data_profile.md"

    results = audit_dataset(dataset_path, expected_sha256=EXPECTED_SHA256)

    # Ensure parent directory for reports exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = format_markdown_report(results)
    report_path.write_text(markdown_content, encoding="utf-8")

    # Format human-readable console summary
    presence_str = "Present" if results.get("file_exists") else "Missing"
    file_size = results.get("file_size_bytes", 0)
    file_size_mb = file_size / (1024 * 1024)

    sha_status = (
        "MATCH"
        if results.get("sha256_match")
        else f"MISMATCH (expected {EXPECTED_SHA256})"
    )

    narratives = results.get("narrative_audit") or {}
    usable_narratives = narratives.get("usable_count", "N/A")
    unusable_narratives = narratives.get("unusable_count", "N/A")
    dup_narratives = narratives.get("duplicate_count", "N/A")

    complaint_ids = results.get("complaint_id_audit") or {}
    missing_ids = complaint_ids.get("missing", "N/A")
    dup_ids = complaint_ids.get("duplicates", "N/A")

    date_range = results.get("date_range") or (None, None)
    min_date, max_date = date_range
    date_str = (
        f"{min_date} to {max_date}"
        if min_date and max_date
        else "N/A"
    )

    overall_status = "PASSED" if results.get("verification_passed") else "FAILED"

    print("=" * 60)
    print("Dataset Verification Summary")
    print("=" * 60)
    print(f"Dataset Presence:           {presence_str} ({dataset_path})")
    print(f"SHA-256 Verification:       {sha_status}")
    print(f"File SHA-256:               {results.get('sha256', 'N/A')}")
    print(f"File Size:                  {file_size:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Total Records:              {results.get('total_rows', 'N/A'):,}" if isinstance(results.get('total_rows'), int) else f"Total Records:              {results.get('total_rows', 'N/A')}")
    print(f"Total Columns:              {results.get('total_columns', 'N/A')}")
    print(
        f"Narrative Availability:     {usable_narratives:,} usable "
        f"({unusable_narratives} unusable, {dup_narratives:,} duplicates)"
        if isinstance(usable_narratives, int)
        else f"Narrative Availability:     {usable_narratives}"
    )
    print(
        f"Duplicate Identifier Count: {dup_ids:,} duplicate IDs ({missing_ids} missing)"
        if isinstance(dup_ids, int)
        else f"Duplicate Identifier Count: {dup_ids}"
    )
    print(f"Date Range:                 {date_str}")
    print(f"Report Generated:           {report_path}")
    print(f"Overall Verification:       {overall_status}")
    print("=" * 60)

    if not results.get("verification_passed"):
        print("Verification failures encountered:")
        for reason in results.get("failure_reasons", []):
            print(f"  - {reason}")
        return 1

    # Check for non-critical warnings
    missing_vals = results.get("missing_values", {})
    total_rows = results.get("total_rows", 0)
    for col, count in missing_vals.items():
        if count == total_rows and total_rows > 0:
            print(f"Warning: Column '{col}' is completely unpopulated ({count:,} missing values).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
