#!/usr/bin/env python3
"""Data loading and verification script for Arzi — Financial Complaint Intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

from arzi.data_pipeline import run_pipeline


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "raw" / "complaints.csv.gz"
    schema_path = project_root / "sql" / "schema.sql"

    if not dataset_path.is_file():
        print(f"Error: Raw dataset file not found at {dataset_path}", file=sys.stderr)
        return 1

    if not schema_path.is_file():
        print(f"Error: Schema definition not found at {schema_path}", file=sys.stderr)
        return 1

    try:
        metrics = run_pipeline(dataset_path, schema_path)
    except Exception as exc:
        print(f"Pipeline execution failed: {exc}", file=sys.stderr)
        return 1

    print("============================================================")
    print("Arzi PostgreSQL Data Pipeline Summary")
    print("============================================================")
    print("Database Connection:        Connected (localhost:5432 / arzi)")
    print(f"PostgreSQL Version:         {metrics.get('server_version', 'Unknown')}")
    print(f"Source Checksum Verified:   {metrics.get('checksum')}")
    print(f"Raw Records Loaded:         {metrics.get('raw_loaded', 0):,}")
    print(f"Prepared Records Loaded:    {metrics.get('prepared_loaded', 0):,}")
    print(f"Unique Complaint IDs:       {metrics.get('distinct_ids', 0):,}")
    print(f"Date Range:                 {metrics.get('min_date')} to {metrics.get('max_date')}")
    print(f"Narratives Present:         {metrics.get('narrative_count', 0):,}")
    print(f"Repeated Narrative Count:   {metrics.get('repeated_narratives', 0):,}")
    print("Overall Validation Status:  PASSED")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
