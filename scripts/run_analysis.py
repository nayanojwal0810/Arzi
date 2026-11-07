#!/usr/bin/env python3
"""Run SQL business analysis and exploratory data visualization for Arzi."""

from __future__ import annotations

import sys
from pathlib import Path

from arzi.analysis import (
    EXPECTED_RECORD_COUNT,
    run_full_analysis,
)
from arzi.database import TARGET_DB, get_connection


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    views_path = project_root / "sql" / "analysis_views.sql"
    eda_output_dir = project_root / "data" / "processed" / "eda"
    summary_output_file = project_root / "data" / "processed" / "analysis_summary.json"

    if not views_path.is_file():
        print(f"Error: Views definition file not found at {views_path}", file=sys.stderr)
        return 1

    # 1. Connect specifically to arzi and verify table presence and count
    try:
        with get_connection(dbname=TARGET_DB) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = 'arzi' AND table_name = 'complaints';
                    """
                )
                if cur.fetchone()[0] == 0:
                    print("Error: Table arzi.complaints does not exist.", file=sys.stderr)
                    return 1

                cur.execute("SELECT COUNT(*) FROM arzi.complaints;")
                count = cur.fetchone()[0]
                if count != EXPECTED_RECORD_COUNT:
                    print(
                        f"Error: arzi.complaints record count mismatch: expected {EXPECTED_RECORD_COUNT:,}, got {count:,}",
                        file=sys.stderr,
                    )
                    return 1

                db_host = conn.info.host or "localhost"
                db_port = conn.info.port or 5432
                db_name = conn.info.dbname or TARGET_DB
    except Exception as exc:
        print(f"Database connection or verification failed: {exc}", file=sys.stderr)
        return 1

    # 2. Run analysis workflow
    try:
        summary = run_full_analysis(
            views_path=views_path,
            eda_output_dir=eda_output_dir,
            summary_output_file=summary_output_file,
        )
    except Exception as exc:
        print(f"Analysis execution failed: {exc}", file=sys.stderr)
        return 1

    # 3. Print concise, professional console summary
    date_info = summary.get("date_range", {})
    catalog = summary.get("catalog_counts", {})
    timely = summary.get("timely_response", {})

    print("============================================================")
    print("Arzi SQL Business Analysis & EDA Summary")
    print("============================================================")
    print(f"Database Connection:        Connected ({db_host}:{db_port} / {db_name})")
    print(f"Complaints Analyzed:        {summary.get('dataset_record_count', 0):,}")
    print(f"Date Range:                 {date_info.get('min_date')} to {date_info.get('max_date')} (Jan 2020: partial)")
    print(f"Catalog Totals:             {catalog.get('products')} products, {catalog.get('issues')} issues, {catalog.get('companies'):,} companies")
    print(f"Usable Narratives:          {catalog.get('usable_narratives', 0):,}")
    print(f"Overall Timely Rate:        {timely.get('timely_response_rate_pct')}%, denominator: {timely.get('known_denominator'):,}")
    print("Top 3 Products by Volume:")
    for p in summary.get("top_five_products", [])[:3]:
        print(f"  - {p['product']}: {p['complaint_count']:,} ({p['share_pct']}%)")
    print("Top 3 Issues by Volume:")
    for iss in summary.get("top_five_issues", [])[:3]:
        print(f"  - {iss['issue']}: {iss['complaint_count']:,}")
    print("Generated Artifacts:")
    print(f"  - Charts: {eda_output_dir}")
    print(f"  - Summary JSON: {summary_output_file}")
    print("Overall Analysis Status:    PASSED")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
