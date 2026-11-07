#!/usr/bin/env python3
"""Run statistical analysis and evidence-based investigation for Arzi."""

from __future__ import annotations

import sys
from pathlib import Path

from arzi.analysis import EXPECTED_RECORD_COUNT
from arzi.database import TARGET_DB, get_connection
from arzi.statistics import run_full_statistical_analysis


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    views_path = project_root / "sql" / "statistical_views.sql"
    output_dir = project_root / "data" / "processed" / "statistics"
    summary_file = project_root / "data" / "processed" / "statistics_summary.json"

    if not views_path.is_file():
        print(f"Error: Views definition file not found at {views_path}", file=sys.stderr)
        return 1

    # 1. Connect and verify complaint data presence and count
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

    # 2. Run statistical workflow
    try:
        summary = run_full_statistical_analysis(
            views_path=views_path,
            output_dir=output_dir,
            summary_file=summary_file,
        )
    except Exception as exc:
        print(f"Statistical analysis execution failed: {exc}", file=sys.stderr)
        return 1

    # 3. Print concise factual console summary
    timeliness = summary.get("timely_response_analysis", {})
    mix = summary.get("product_mix_shift_analysis", {})
    chi2_info = mix.get("chi_square", {})
    concentration = summary.get("issue_concentration_analysis", [])

    print("============================================================")
    print("Arzi Statistical Analysis & Evidence Summary")
    print("============================================================")
    print(f"Database Connection:        Connected ({db_host}:{db_port} / {db_name})")
    print(f"Dataset Complaints Analyzed:{summary.get('dataset_record_count', 0):,}")
    print(f"Overall Timely-Response Rate:{timeliness.get('overall_timely_rate_pct')}%, denominator: {timeliness.get('overall_known_denominator', 0):,}")
    print(f"Product Timeliness Evaluated:{len(timeliness.get('evaluations', []))} products")
    print(f"Products Meeting Timeliness:{len(timeliness.get('flagged_signals', []))} flagged investigation signals")
    print(f"2019 Half-Year Record Counts:H1 = {mix.get('h1_total', 0):,}, H2 = {mix.get('h2_total', 0):,}")
    print(f"Product-Mix Independence:   Chi2 = {chi2_info.get('statistic')}, p = {chi2_info.get('p_value'):.2e}, Cramer's V = {chi2_info.get('cramers_v')}")
    print(f"Products Meeting Mix Shift: {len(mix.get('flagged_signals', []))} flagged mix signals")
    print(f"Issue Concentrations Built: {len(concentration)} product profiles")
    print("Generated Artifacts:")
    print(f"  - Charts: {output_dir}")
    print(f"  - Summary JSON: {summary_file}")
    print("Overall Completion Status:  PASSED")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
