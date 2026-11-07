"""SQL business analysis and exploratory data visualization for Arzi."""

from __future__ import annotations

import decimal
import json
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from arzi.database import (
    TARGET_DB,
    apply_schema,
    get_connection,
)

EXPECTED_RECORD_COUNT = 117214
EXPECTED_MIN_DATE = "2019-01-01"
EXPECTED_MAX_DATE = "2020-01-09"


def serialize_json_value(obj: Any) -> Any:
    """JSON serialization helper for date, datetime, and Decimal types."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def calculate_share_pct(count: int, total: int) -> float | None:
    """Calculate percentage share safely, returning None if total is zero or negative."""
    if total <= 0:
        return None
    return round((count * 100.0) / total, 2)


def check_partial_month(month_date: date | str, max_dataset_date: date | str) -> bool:
    """Determine whether a month is partial by checking if max date is before month end."""
    if isinstance(month_date, str):
        month_date = date.fromisoformat(month_date)
    if isinstance(max_dataset_date, str):
        max_dataset_date = date.fromisoformat(max_dataset_date)

    if (month_date.year, month_date.month) == (max_dataset_date.year, max_dataset_date.month):
        next_month_year = month_date.year + (1 if month_date.month == 12 else 0)
        next_month_month = 1 if month_date.month == 12 else month_date.month + 1
        first_of_next = date(next_month_year, next_month_month, 1)
        month_end = date.fromordinal(first_of_next.toordinal() - 1)
        return max_dataset_date < month_end
    return False


def apply_analysis_views(views_path: Path | str) -> None:
    """Execute analysis view creation SQL statements on the target database."""
    with get_connection(dbname=TARGET_DB) as conn:
        apply_schema(conn, views_path)
        conn.commit()


def fetch_product_summary() -> list[dict[str, Any]]:
    """Retrieve product-level aggregate statistics from arzi.v_product_summary."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    product,
                    complaint_count,
                    complaint_share_pct,
                    distinct_issue_count,
                    distinct_company_count,
                    narrative_count,
                    known_timely_response_count,
                    timely_response_rate_pct
                FROM arzi.v_product_summary
                ORDER BY complaint_count DESC, product ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_monthly_trend() -> list[dict[str, Any]]:
    """Retrieve continuous monthly complaint trend from arzi.v_monthly_trend."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    month_start,
                    complaint_count,
                    previous_month_count,
                    month_over_month_change,
                    month_over_month_change_pct,
                    is_partial_last_month
                FROM arzi.v_monthly_trend
                ORDER BY month_start ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_monthly_product_trend() -> list[dict[str, Any]]:
    """Retrieve monthly counts grouped by financial product from arzi.v_monthly_product_trend."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    month_start,
                    product,
                    complaint_count
                FROM arzi.v_monthly_product_trend
                ORDER BY month_start ASC, product ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_issue_summary() -> list[dict[str, Any]]:
    """Retrieve product-specific issue distribution from arzi.v_issue_summary."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    product,
                    issue,
                    complaint_count,
                    product_complaint_share_pct
                FROM arzi.v_issue_summary
                ORDER BY product ASC, complaint_count DESC, issue ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_top_issues_overall(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve top issues overall aggregated across all products."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    issue,
                    SUM(complaint_count) AS complaint_count
                FROM arzi.v_issue_summary
                GROUP BY issue
                ORDER BY complaint_count DESC, issue ASC
                LIMIT %s;
                """,
                (limit,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_company_summary(limit: int | None = None) -> list[dict[str, Any]]:
    """Retrieve company-level complaint counts and timely response behavior."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            query = """
                SELECT
                    company,
                    complaint_count,
                    distinct_product_count,
                    known_timely_response_count,
                    timely_response_rate_pct
                FROM arzi.v_company_summary
                ORDER BY complaint_count DESC, company ASC
            """
            if limit:
                query += " LIMIT %s"
                cur.execute(query, (limit,))
            else:
                cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_response_summary() -> list[dict[str, Any]]:
    """Retrieve company response outcomes by product from arzi.v_response_summary."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    product,
                    company_response_to_consumer,
                    complaint_count,
                    product_response_share_pct
                FROM arzi.v_response_summary
                ORDER BY product ASC, complaint_count DESC, company_response_to_consumer ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_channel_summary() -> list[dict[str, Any]]:
    """Retrieve complaint submission channel distribution from arzi.v_channel_summary."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    submitted_via,
                    complaint_count,
                    complaint_share_pct
                FROM arzi.v_channel_summary
                ORDER BY complaint_count DESC, submitted_via ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def validate_analysis_metrics(
    product_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    monthly_product_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Validate cross-view record and percentage consistency against acceptance rules."""
    errors: list[str] = []

    # 1. Total record consistency across groupings
    product_total = sum(int(r["complaint_count"]) for r in product_rows)
    monthly_total = sum(int(r["complaint_count"]) for r in monthly_rows)
    issue_total = sum(int(r["complaint_count"]) for r in issue_rows)
    response_total = sum(int(r["complaint_count"]) for r in response_rows)
    channel_total = sum(int(r["complaint_count"]) for r in channel_rows)

    check_list = [
        ("product summary", product_total),
        ("monthly trend", monthly_total),
        ("issue summary", issue_total),
        ("response summary", response_total),
        ("channel summary", channel_total),
    ]

    if monthly_product_rows is not None:
        monthly_prod_total = sum(int(r["complaint_count"]) for r in monthly_product_rows)
        check_list.append(("monthly product trend", monthly_prod_total))

    for name, total in check_list:
        if total != EXPECTED_RECORD_COUNT:
            errors.append(
                f"{name} count mismatch: expected {EXPECTED_RECORD_COUNT:,}, got {total:,}"
            )

    # 2. Monthly trend checks
    if monthly_rows:
        last_month = monthly_rows[-1]
        if not last_month.get("is_partial_last_month"):
            errors.append("Final month (January 2020) was not flagged as is_partial_last_month")
        if last_month.get("month_over_month_change_pct") is not None:
            errors.append(
                "Partial last month must have NULL month_over_month_change_pct"
            )
        if last_month.get("month_over_month_change") is not None:
            errors.append(
                "Partial last month must have NULL month_over_month_change"
            )

    # 3. Product-level count reconciliations
    product_counts = {r["product"]: int(r["complaint_count"]) for r in product_rows}

    # Sum of issues per product = product complaint count
    issues_by_prod: dict[str, int] = {}
    issue_shares_by_prod: dict[str, list[float]] = {}
    for r in issue_rows:
        prod = r["product"]
        issues_by_prod[prod] = issues_by_prod.get(prod, 0) + int(r["complaint_count"])
        if r.get("product_complaint_share_pct") is not None:
            issue_shares_by_prod.setdefault(prod, []).append(float(r["product_complaint_share_pct"]))

    for prod, expected_cnt in product_counts.items():
        actual_cnt = issues_by_prod.get(prod, 0)
        if actual_cnt != expected_cnt:
            errors.append(
                f"Product '{prod}' issue count sum ({actual_cnt:,}) does not match product count ({expected_cnt:,})"
            )
        shares = issue_shares_by_prod.get(prod, [])
        if shares:
            share_sum = sum(shares)
            tolerance = len(shares) * 0.005 + 0.02
            if abs(share_sum - 100.0) > tolerance:
                errors.append(
                    f"Product '{prod}' issue shares sum ({share_sum:.2f}%) exceeds tolerance ({tolerance:.3f}%)"
                )

    # Sum of responses per product = product complaint count
    responses_by_prod: dict[str, int] = {}
    response_shares_by_prod: dict[str, list[float]] = {}
    for r in response_rows:
        prod = r["product"]
        responses_by_prod[prod] = responses_by_prod.get(prod, 0) + int(r["complaint_count"])
        if r.get("product_response_share_pct") is not None:
            response_shares_by_prod.setdefault(prod, []).append(float(r["product_response_share_pct"]))

    for prod, expected_cnt in product_counts.items():
        actual_cnt = responses_by_prod.get(prod, 0)
        if actual_cnt != expected_cnt:
            errors.append(
                f"Product '{prod}' response category count sum ({actual_cnt:,}) does not match product count ({expected_cnt:,})"
            )
        shares = response_shares_by_prod.get(prod, [])
        if shares:
            share_sum = sum(shares)
            tolerance = len(shares) * 0.005 + 0.02
            if abs(share_sum - 100.0) > tolerance:
                errors.append(
                    f"Product '{prod}' response shares sum ({share_sum:.2f}%) exceeds tolerance ({tolerance:.3f}%)"
                )

    # Sum of monthly product trend per product = product complaint count
    if monthly_product_rows is not None:
        monthly_by_prod: dict[str, int] = {}
        for r in monthly_product_rows:
            prod = r["product"]
            monthly_by_prod[prod] = monthly_by_prod.get(prod, 0) + int(r["complaint_count"])
        for prod, expected_cnt in product_counts.items():
            actual_cnt = monthly_by_prod.get(prod, 0)
            if actual_cnt != expected_cnt:
                errors.append(
                    f"Product '{prod}' monthly trend sum ({actual_cnt:,}) does not match product count ({expected_cnt:,})"
                )

    # 4. Overall percentage sum checks (~100%)
    product_share_sum = sum(float(r["complaint_share_pct"]) for r in product_rows)
    prod_tolerance = len(product_rows) * 0.005 + 0.02
    if abs(product_share_sum - 100.0) > prod_tolerance:
        errors.append(f"Product share percentage sum out of bounds: {product_share_sum:.2f}%")

    channel_share_sum = sum(float(r["complaint_share_pct"]) for r in channel_rows)
    chan_tolerance = len(channel_rows) * 0.005 + 0.02
    if abs(channel_share_sum - 100.0) > chan_tolerance:
        errors.append(f"Channel share percentage sum out of bounds: {channel_share_sum:.2f}%")

    # 5. Timely response checks
    for r in product_rows:
        known = int(r.get("known_timely_response_count", 0))
        cnt = int(r["complaint_count"])
        if known > cnt:
            errors.append(f"Known timely response count ({known}) exceeds complaint count ({cnt}) for {r['product']}")
        rate = r.get("timely_response_rate_pct")
        if known > 0 and rate is not None and not (0.0 <= float(rate) <= 100.0):
            errors.append(f"Timely response rate {rate}% out of bounds for {r['product']}")

    if errors:
        raise ValueError("Analysis metrics validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def build_analysis_summary(
    product_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    company_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile verified findings into a structured analysis dictionary."""
    total_complaints = sum(int(r["complaint_count"]) for r in product_rows)
    total_narratives = sum(int(r["narrative_count"]) for r in product_rows)
    distinct_products = len(product_rows)

    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(date_received), MAX(date_received) FROM arzi.complaints;")
            min_date, max_date = cur.fetchone()

            cur.execute("SELECT COUNT(DISTINCT issue), COUNT(DISTINCT company) FROM arzi.complaints;")
            distinct_issues, distinct_companies = cur.fetchone()

            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE timely_response IS NOT NULL) AS known_denominator,
                    COUNT(*) FILTER (WHERE timely_response = TRUE) AS timely_count,
                    COUNT(*) FILTER (WHERE timely_response = FALSE) AS untimely_count,
                    COUNT(*) FILTER (WHERE timely_response IS NULL) AS null_count,
                    ROUND((COUNT(*) FILTER (WHERE timely_response = TRUE) * 100.0 / NULLIF(COUNT(*) FILTER (WHERE timely_response IS NOT NULL), 0)), 2) AS timely_rate
                FROM arzi.complaints;
                """
            )
            known_den, timely_cnt, untimely_cnt, null_cnt, timely_rate = cur.fetchone()

            cur.execute(
                """
                SELECT company_response_to_consumer, COUNT(*) AS count
                FROM arzi.complaints
                GROUP BY company_response_to_consumer
                ORDER BY count DESC;
                """
            )
            response_categories = [{"category": row[0], "count": row[1]} for row in cur.fetchall()]

    top_five_products = [
        {"product": r["product"], "complaint_count": int(r["complaint_count"]), "share_pct": float(r["complaint_share_pct"])}
        for r in sorted(product_rows, key=lambda x: int(x["complaint_count"]), reverse=True)[:5]
    ]

    top_five_issues = fetch_top_issues_overall(limit=5)
    top_five_issues_formatted = [
        {"issue": r["issue"], "complaint_count": int(r["complaint_count"])}
        for r in top_five_issues
    ]

    top_five_companies = [
        {
            "company": r["company"],
            "complaint_count": int(r["complaint_count"]),
            "timely_response_rate_pct": float(r["timely_response_rate_pct"]) if r.get("timely_response_rate_pct") is not None else None,
            "observation_note": "Volume observation within published complaint records",
        }
        for r in company_rows[:5]
    ]

    monthly_summary = [
        {
            "month_start": serialize_json_value(r["month_start"]),
            "complaint_count": int(r["complaint_count"]),
            "month_over_month_change_pct": float(r["month_over_month_change_pct"]) if r["month_over_month_change_pct"] is not None else None,
            "is_partial_last_month": bool(r["is_partial_last_month"]),
        }
        for r in monthly_rows
    ]

    return {
        "dataset_record_count": total_complaints,
        "date_range": {
            "min_date": serialize_json_value(min_date),
            "max_date": serialize_json_value(max_date),
            "is_partial_final_month": True,
            "final_month_note": "January 2020 contains records through 2020-01-09 only",
        },
        "catalog_counts": {
            "products": distinct_products,
            "issues": distinct_issues,
            "companies": distinct_companies,
            "usable_narratives": total_narratives,
        },
        "timely_response": {
            "known_denominator": known_den,
            "timely_count": timely_cnt,
            "untimely_count": untimely_cnt,
            "null_count": null_cnt,
            "timely_response_rate_pct": float(timely_rate) if timely_rate is not None else None,
        },
        "top_five_products": top_five_products,
        "top_five_issues": top_five_issues_formatted,
        "top_five_companies_by_volume": top_five_companies,
        "monthly_trend": monthly_summary,
        "company_response_categories": response_categories,
        "channel_distribution": [
            {"channel": r["submitted_via"], "count": int(r["complaint_count"]), "share_pct": float(r["complaint_share_pct"])}
            for r in channel_rows
        ],
    }


def generate_charts(
    monthly_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    top_issues_rows: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Generate and save the three required exploratory charts using matplotlib."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    # Chart A: Monthly Complaint Volume
    chart_a_path = out_dir / "monthly_complaints.png"
    months = [str(r["month_start"])[:7] for r in monthly_rows]
    counts = [int(r["complaint_count"]) for r in monthly_rows]
    is_partial = [bool(r.get("is_partial_last_month")) for r in monthly_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#d9534f" if p else "#337ab7" for p in is_partial]
    bars = ax.bar(months, counts, color=bar_colors, edgecolor="black", linewidth=0.8)

    ax.set_title("Monthly Complaint Volume (2019-01 through 2020-01)", fontsize=13, pad=12)
    ax.set_xlabel("Month Received", fontsize=11)
    ax.set_ylabel("Complaint Count", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, partial, count in zip(bars, is_partial, counts):
        if partial:
            ax.annotate(
                f"{count:,}\n(Partial\nthrough Jan 9)",
                xy=(bar.get_x() + bar.get_width() / 2, count),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#a94442",
                weight="bold",
            )
        else:
            ax.annotate(
                f"{count:,}",
                xy=(bar.get_x() + bar.get_width() / 2, count),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    y_max = max(counts) * 1.15
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(chart_a_path, dpi=150)
    plt.close(fig)
    generated_files.append(chart_a_path)

    # Chart B: Complaints by Financial Product
    chart_b_path = out_dir / "product_distribution.png"
    sorted_products = sorted(product_rows, key=lambda x: int(x["complaint_count"]))
    prod_names = [textwrap.fill(r["product"], width=35) for r in sorted_products]
    prod_counts = [int(r["complaint_count"]) for r in sorted_products]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(prod_names, prod_counts, color="#2e6da4", edgecolor="black", linewidth=0.8)
    ax.set_title("Complaints by Financial Product", fontsize=13, pad=12)
    ax.set_xlabel("Complaint Count", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, count in zip(bars, prod_counts):
        ax.annotate(
            f" {count:,}",
            xy=(count, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, max(prod_counts) * 1.15)
    fig.tight_layout()
    fig.savefig(chart_b_path, dpi=150)
    plt.close(fig)
    generated_files.append(chart_b_path)

    # Chart C: Most Common Complaint Issues
    chart_c_path = out_dir / "top_issues.png"
    sorted_issues = sorted(top_issues_rows, key=lambda x: int(x["complaint_count"]))
    issue_names = [textwrap.fill(r["issue"], width=35) for r in sorted_issues]
    issue_counts = [int(r["complaint_count"]) for r in sorted_issues]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(issue_names, issue_counts, color="#5cb85c", edgecolor="black", linewidth=0.8)
    ax.set_title("Top 10 Most Frequent Complaint Issues", fontsize=13, pad=12)
    ax.set_xlabel("Complaint Count", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, count in zip(bars, issue_counts):
        ax.annotate(
            f" {count:,}",
            xy=(count, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, max(issue_counts) * 1.15)
    fig.tight_layout()
    fig.savefig(chart_c_path, dpi=150)
    plt.close(fig)
    generated_files.append(chart_c_path)

    return generated_files


def run_full_analysis(
    views_path: Path | str,
    eda_output_dir: Path | str,
    summary_output_file: Path | str,
) -> dict[str, Any]:
    """Execute complete analysis pipeline: apply views, validate metrics, render charts, save JSON."""
    v_path = Path(views_path)
    charts_dir = Path(eda_output_dir)
    json_path = Path(summary_output_file)

    # 1. Apply analytical views
    apply_analysis_views(v_path)

    # 2. Fetch results from views
    products = fetch_product_summary()
    monthly = fetch_monthly_trend()
    monthly_products = fetch_monthly_product_trend()
    issues = fetch_issue_summary()
    top_issues = fetch_top_issues_overall(limit=10)
    companies = fetch_company_summary(limit=10)
    responses = fetch_response_summary()
    channels = fetch_channel_summary()

    # 3. Validate metrics consistency
    validate_analysis_metrics(
        product_rows=products,
        monthly_rows=monthly,
        issue_rows=issues,
        response_rows=responses,
        channel_rows=channels,
        monthly_product_rows=monthly_products,
    )

    # 4. Generate EDA charts
    generate_charts(monthly, products, top_issues, charts_dir)

    # 5. Compile and save structured JSON summary
    summary_data = build_analysis_summary(
        products, monthly, issues, companies, responses, channels
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary_data, indent=2, default=serialize_json_value),
        encoding="utf-8",
    )

    return summary_data
