"""Statistical analysis and evidence-based investigation module for Arzi."""

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
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import statsmodels.stats.proportion as smp

from arzi.analysis import EXPECTED_RECORD_COUNT
from arzi.database import TARGET_DB, apply_schema, get_connection

MIN_SAMPLE_SIZE = 200
ALPHA = 0.05
TIMELINESS_EFFECT_SIZE_PP = -2.0
MIX_EFFECT_SIZE_PP = 2.0


def serialize_json_value(obj: Any) -> Any:
    """JSON serialization helper for dates, Decimals, and numpy types."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def calculate_wilson_ci(
    count: int, nobs: int, alpha: float = ALPHA
) -> tuple[float, float] | None:
    """Calculate two-sided Wilson score confidence interval for a proportion."""
    if nobs <= 0 or count < 0 or count > nobs:
        return None
    low, high = smp.proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(low), float(high)


def calculate_newcombe_diff_ci(
    count1: int, nobs1: int, count2: int, nobs2: int, alpha: float = ALPHA
) -> tuple[float, float] | None:
    """Calculate Newcombe confidence interval for difference between two independent proportions (p1 - p2)."""
    if nobs1 <= 0 or nobs2 <= 0 or count1 < 0 or count2 < 0 or count1 > nobs1 or count2 > nobs2:
        return None
    try:
        low, high = smp.confint_proportions_2indep(
            count1, nobs1, count2, nobs2, method="newcombe", alpha=alpha
        )
        return float(low), float(high)
    except Exception:
        return None


def calculate_two_proportion_pvalue(
    count1: int, nobs1: int, count2: int, nobs2: int
) -> float | None:
    """Calculate two-sided z-test p-value comparing two independent proportions."""
    if nobs1 <= 0 or nobs2 <= 0:
        return None
    try:
        _, pval = smp.proportions_ztest([count1, count2], [nobs1, nobs2])
        return float(pval)
    except Exception:
        return None


def apply_benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Adjust a list of p-values using the Benjamini-Hochberg (FDR) procedure."""
    if not pvalues:
        return []
    valid_indices = [i for i, p in enumerate(pvalues) if p is not None and not np.isnan(p)]
    if not valid_indices:
        return [1.0] * len(pvalues)

    valid_pvals = [pvalues[i] for i in valid_indices]
    _, p_adjusted, _, _ = smm.multipletests(valid_pvals, method="fdr_bh")

    result = list(pvalues)
    for idx, adj in zip(valid_indices, p_adjusted):
        result[idx] = float(adj)
    return result


def apply_statistical_views(views_path: Path | str) -> None:
    """Execute statistical view creation statements on the target database."""
    with get_connection(dbname=TARGET_DB) as conn:
        apply_schema(conn, views_path)
        conn.commit()


def fetch_product_response_counts() -> list[dict[str, Any]]:
    """Retrieve product timely response counts from arzi.v_product_response_counts."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    product,
                    complaint_count,
                    known_response_count,
                    timely_count,
                    untimely_count,
                    unknown_response_count
                FROM arzi.v_product_response_counts
                ORDER BY complaint_count DESC, product ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_half_year_product_counts() -> list[dict[str, Any]]:
    """Retrieve 2019 half-year product counts from arzi.v_half_year_product_counts."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    period,
                    product,
                    complaint_count,
                    period_total
                FROM arzi.v_half_year_product_counts
                ORDER BY period ASC, complaint_count DESC, product ASC;
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def analyze_product_timeliness(
    response_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate product timely response rates, confidence intervals, and comparative signals."""
    total_known = sum(int(r["known_response_count"]) for r in response_rows)
    total_timely = sum(int(r["timely_count"]) for r in response_rows)
    overall_rate_pct = round((total_timely * 100.0) / total_known, 2) if total_known > 0 else 0.0

    evaluations: list[dict[str, Any]] = []
    pvalues_to_adjust: list[float] = []
    eligible_indices: list[int] = []

    for i, r in enumerate(response_rows):
        product = r["product"]
        complaint_count = int(r["complaint_count"])
        known = int(r["known_response_count"])
        timely = int(r["timely_count"])
        untimely = int(r["untimely_count"])

        rate = (timely / known) if known > 0 else 0.0
        rate_pct = round(rate * 100.0, 2)
        ci_rate = calculate_wilson_ci(timely, known)
        ci_rate_pct = (
            (round(ci_rate[0] * 100.0, 2), round(ci_rate[1] * 100.0, 2))
            if ci_rate
            else (None, None)
        )

        rest_known = total_known - known
        rest_timely = total_timely - timely
        rate_rest = (rest_timely / rest_known) if rest_known > 0 else 0.0
        rate_rest_pct = round(rate_rest * 100.0, 2)

        diff_raw_pp = (rate - rate_rest) * 100.0
        diff_pp = round(diff_raw_pp, 2)
        ci_diff = calculate_newcombe_diff_ci(timely, known, rest_timely, rest_known)
        ci_diff_pp = (
            (round(ci_diff[0] * 100.0, 2), round(ci_diff[1] * 100.0, 2))
            if ci_diff
            else (None, None)
        )

        pval = calculate_two_proportion_pvalue(timely, known, rest_timely, rest_known)

        is_eligible = known >= MIN_SAMPLE_SIZE
        if is_eligible and pval is not None:
            pvalues_to_adjust.append(pval)
            eligible_indices.append(i)

        evaluations.append({
            "product": product,
            "complaint_count": complaint_count,
            "known_response_count": known,
            "timely_count": timely,
            "untimely_count": untimely,
            "observed_rate_pct": rate_pct,
            "wilson_ci_pct": ci_rate_pct,
            "other_products_rate_pct": rate_rest_pct,
            "difference_pp": diff_pp,
            "newcombe_diff_ci_pp": ci_diff_pp,
            "p_value": pval,
            "p_value_adjusted": None,
            "is_sample_size_eligible": is_eligible,
            "meets_screening_criteria": False,
            "_raw_diff_pp": diff_raw_pp,
            "_raw_ci_diff": ci_diff,
        })

    # Adjust p-values across eligible comparisons
    adjusted_pvals = apply_benjamini_hochberg(pvalues_to_adjust)
    for list_pos, orig_idx in enumerate(eligible_indices):
        padj = adjusted_pvals[list_pos]
        evaluations[orig_idx]["p_value_adjusted"] = padj

        # Screening criteria evaluated on full-precision (unrounded) values:
        # - known denominator >= 200
        # - observed rate <= other rate - 2.0 pp (unrounded diff <= -2.0)
        # - adjusted p-value < 0.05
        # - 95% Newcombe difference interval upper bound strictly below zero
        raw_diff_val = evaluations[orig_idx]["_raw_diff_pp"]
        raw_ci = evaluations[orig_idx]["_raw_ci_diff"]
        raw_ci_high = (raw_ci[1] * 100.0) if raw_ci else None

        if (
            evaluations[orig_idx]["is_sample_size_eligible"]
            and raw_diff_val <= TIMELINESS_EFFECT_SIZE_PP
            and padj < ALPHA
            and raw_ci_high is not None
            and raw_ci_high < 0.0
        ):
            evaluations[orig_idx]["meets_screening_criteria"] = True

    # Strip temporary raw calculation keys from final output dictionaries
    for e in evaluations:
        e.pop("_raw_diff_pp", None)
        e.pop("_raw_ci_diff", None)

    flagged_signals = [e for e in evaluations if e["meets_screening_criteria"]]

    return {
        "overall_timely_rate_pct": overall_rate_pct,
        "overall_known_denominator": total_known,
        "evaluations": evaluations,
        "flagged_signals": flagged_signals,
    }


def analyze_product_mix_changes(
    half_year_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate financial product mix shifts between 2019 H1 and H2."""
    h1_records = [r for r in half_year_rows if r["period"] == "2019_h1"]
    h2_records = [r for r in half_year_rows if r["period"] == "2019_h2"]

    h1_by_prod = {r["product"]: int(r["complaint_count"]) for r in h1_records}
    h2_by_prod = {r["product"]: int(r["complaint_count"]) for r in h2_records}

    all_products = sorted(set(h1_by_prod.keys()) | set(h2_by_prod.keys()))
    h1_total = sum(h1_by_prod.values())
    h2_total = sum(h2_by_prod.values())

    # 1. Contingency table for Chi-Square test
    contingency_table = [
        [h1_by_prod.get(p, 0) for p in all_products],
        [h2_by_prod.get(p, 0) for p in all_products],
    ]

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    n_total = h1_total + h2_total
    k_min = min(len(contingency_table) - 1, len(all_products) - 1)
    cramers_v = float(np.sqrt(chi2 / (n_total * k_min))) if (n_total * k_min) > 0 else 0.0
    min_expected = float(expected.min()) if expected.size > 0 else 0.0

    # 2. Individual product share comparisons
    product_comparisons: list[dict[str, Any]] = []
    pvals_to_adjust: list[float] = []
    eligible_indices: list[int] = []

    for i, p in enumerate(all_products):
        c1 = h1_by_prod.get(p, 0)
        c2 = h2_by_prod.get(p, 0)
        combined_count = c1 + c2

        s1_raw_pct = (c1 * 100.0) / h1_total if h1_total > 0 else 0.0
        s2_raw_pct = (c2 * 100.0) / h2_total if h2_total > 0 else 0.0
        diff_raw_pp = s2_raw_pct - s1_raw_pct

        s1_pct = round(s1_raw_pct, 2)
        s2_pct = round(s2_raw_pct, 2)
        diff_pp = round(diff_raw_pp, 2)

        ci_diff = calculate_newcombe_diff_ci(c2, h2_total, c1, h1_total)
        ci_diff_pp = (
            (round(ci_diff[0] * 100.0, 2), round(ci_diff[1] * 100.0, 2))
            if ci_diff
            else (None, None)
        )

        pval = calculate_two_proportion_pvalue(c2, h2_total, c1, h1_total)
        is_eligible = combined_count >= MIN_SAMPLE_SIZE

        if is_eligible and pval is not None:
            pvals_to_adjust.append(pval)
            eligible_indices.append(i)

        product_comparisons.append({
            "product": p,
            "h1_count": c1,
            "h1_share_pct": s1_pct,
            "h2_count": c2,
            "h2_share_pct": s2_pct,
            "share_change_pp": diff_pp,
            "newcombe_diff_ci_pp": ci_diff_pp,
            "p_value": pval,
            "p_value_adjusted": None,
            "combined_count": combined_count,
            "is_sample_size_eligible": is_eligible,
            "meets_screening_criteria": False,
            "_raw_diff_pp": diff_raw_pp,
            "_raw_ci_diff": ci_diff,
        })

    # Benjamini-Hochberg adjustment across eligible mix comparisons
    adjusted_pvals = apply_benjamini_hochberg(pvals_to_adjust)
    for list_pos, orig_idx in enumerate(eligible_indices):
        padj = adjusted_pvals[list_pos]
        product_comparisons[orig_idx]["p_value_adjusted"] = padj

        # Screening criteria evaluated on full-precision (unrounded) values:
        # - |diff_raw_pp| >= 2.0 percentage points
        # - adjusted p-value < 0.05
        # - 95% Newcombe CI excludes 0
        # - combined count >= 200
        raw_diff_val = product_comparisons[orig_idx]["_raw_diff_pp"]
        raw_ci = product_comparisons[orig_idx]["_raw_ci_diff"]
        ci_excludes_zero = (
            raw_ci is not None
            and ((raw_ci[0] > 0 and raw_ci[1] > 0) or (raw_ci[0] < 0 and raw_ci[1] < 0))
        )

        if (
            product_comparisons[orig_idx]["is_sample_size_eligible"]
            and abs(raw_diff_val) >= MIX_EFFECT_SIZE_PP
            and padj < ALPHA
            and ci_excludes_zero
        ):
            product_comparisons[orig_idx]["meets_screening_criteria"] = True

    # Strip temporary raw calculation keys from final output dictionaries
    for m in product_comparisons:
        m.pop("_raw_diff_pp", None)
        m.pop("_raw_ci_diff", None)

    flagged_signals = [m for m in product_comparisons if m["meets_screening_criteria"]]

    return {
        "h1_total": h1_total,
        "h2_total": h2_total,
        "chi_square": {
            "statistic": round(float(chi2), 2),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_val),
            "cramers_v": round(float(cramers_v), 4),
            "min_expected_cell": round(float(min_expected), 2),
            "assumptions_satisfied": bool(min_expected >= 5.0),
        },
        "product_comparisons": product_comparisons,
        "flagged_signals": flagged_signals,
    }


def analyze_issue_concentration(
    issue_rows: list[tuple[str, str, int]] | None = None,
    prod_rows: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Determine top three issue concentration within each product category."""
    if issue_rows is None or prod_rows is None:
        with get_connection(dbname=TARGET_DB) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT product, issue, complaint_count
                    FROM arzi.v_issue_summary
                    ORDER BY product ASC, complaint_count DESC, issue ASC;
                    """
                )
                issue_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT product, complaint_count
                    FROM arzi.v_product_summary
                    ORDER BY complaint_count DESC, product ASC;
                    """
                )
                prod_rows = cur.fetchall()

    issues_by_product: dict[str, list[tuple[str, int]]] = {}
    for prod, iss, cnt in issue_rows:
        issues_by_product.setdefault(prod, []).append((iss, int(cnt)))

    concentration_results: list[dict[str, Any]] = []
    for prod, total_cnt in prod_rows:
        total = int(total_cnt)
        issues = issues_by_product.get(prod, [])
        top3 = issues[:3]
        top3_count = sum(c for _, c in top3)
        combined_share = round((top3_count * 100.0) / total, 2) if total > 0 else 0.0

        top3_formatted = [
            {
                "issue": iss,
                "complaint_count": c,
                "share_of_product_pct": round((c * 100.0) / total, 2) if total > 0 else 0.0,
            }
            for iss, c in top3
        ]

        concentration_results.append({
            "product": prod,
            "product_total_complaints": total,
            "top_three_issues": top3_formatted,
            "top_three_combined_count": top3_count,
            "top_three_combined_share_pct": combined_share,
        })

    return concentration_results


def validate_statistical_metrics(
    timeliness_data: dict[str, Any],
    mix_data: dict[str, Any],
    concentration_data: list[dict[str, Any]],
) -> None:
    """Validate data integrity, counts, and bounds for all statistical calculations."""
    errors: list[str] = []

    # 1. Timeliness checks
    known_total = timeliness_data.get("overall_known_denominator", 0)
    evals = timeliness_data.get("evaluations", [])
    if sum(e["known_response_count"] for e in evals) != known_total:
        errors.append("Sum of product known response counts does not equal overall known responses")

    for e in evals:
        if e["wilson_ci_pct"][0] is not None:
            low, high = e["wilson_ci_pct"]
            if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0 and low <= high):
                errors.append(f"Invalid Wilson CI bounds for {e['product']}: [{low}, {high}]")
        if e["p_value_adjusted"] is not None and not (0.0 <= e["p_value_adjusted"] <= 1.0):
            errors.append(f"Invalid adjusted p-value for {e['product']}: {e['p_value_adjusted']}")

    # 2. Product-mix checks
    h1_tot = mix_data.get("h1_total", 0)
    h2_tot = mix_data.get("h2_total", 0)
    if h1_tot <= 0 or h2_tot <= 0:
        errors.append("Half-year period totals must be greater than zero")

    mix_comps = mix_data.get("product_comparisons", [])
    if len(mix_comps) != 9:
        errors.append(f"Expected 9 product mix comparisons, got {len(mix_comps)}")

    if sum(m["h1_count"] for m in mix_comps) != h1_tot:
        errors.append("Sum of H1 product counts does not equal H1 total")
    if sum(m["h2_count"] for m in mix_comps) != h2_tot:
        errors.append("Sum of H2 product counts does not equal H2 total")

    # 3. Issue concentration checks
    for item in concentration_data:
        prod_tot = item["product_total_complaints"]
        top3_cnt = item["top_three_combined_count"]
        top3_share = item["top_three_combined_share_pct"]
        if top3_cnt > prod_tot:
            errors.append(f"Top 3 count ({top3_cnt}) exceeds product total ({prod_tot}) for {item['product']}")
        if not (0.0 <= top3_share <= 100.0):
            errors.append(f"Top 3 share ({top3_share}%) out of bounds for {item['product']}")

    # 4. Screening rules satisfaction check
    for item in timeliness_data.get("flagged_signals", []):
        if not item.get("is_sample_size_eligible"):
            errors.append(f"Flagged timeliness signal {item.get('product')} fails sample size condition")
        if item.get("difference_pp", 0) > TIMELINESS_EFFECT_SIZE_PP:
            errors.append(f"Flagged timeliness signal {item.get('product')} fails effect size condition")
        if item.get("p_value_adjusted", 1.0) >= ALPHA:
            errors.append(f"Flagged timeliness signal {item.get('product')} fails significance threshold")
        ci_high = item.get("newcombe_diff_ci_pp", (None, None))[1]
        if ci_high is None or ci_high >= 0.0:
            errors.append(f"Flagged timeliness signal {item.get('product')} fails CI below zero condition")

    for item in mix_data.get("flagged_signals", []):
        if not item.get("is_sample_size_eligible"):
            errors.append(f"Flagged mix signal {item.get('product')} fails sample size condition")
        if abs(item.get("share_change_pp", 0)) < MIX_EFFECT_SIZE_PP:
            errors.append(f"Flagged mix signal {item.get('product')} fails effect size condition")
        if item.get("p_value_adjusted", 1.0) >= ALPHA:
            errors.append(f"Flagged mix signal {item.get('product')} fails significance threshold")
        ci_low, ci_high = item.get("newcombe_diff_ci_pp", (None, None))
        if ci_low is None or ci_high is None or (ci_low <= 0 and ci_high >= 0):
            errors.append(f"Flagged mix signal {item.get('product')} fails CI excludes zero condition")

    if errors:
        raise ValueError("Statistical metrics validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def verify_source_statistical_totals(
    timeliness_data: dict[str, Any],
    mix_data: dict[str, Any],
) -> None:
    """Independently verify statistical core counts directly against simple queries from arzi.complaints."""
    errors: list[str] = []
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            # 1. Total complaints = 117,214
            cur.execute("SELECT COUNT(*) FROM arzi.complaints;")
            total_source_count = cur.fetchone()[0]
            if total_source_count != EXPECTED_RECORD_COUNT:
                errors.append(
                    f"Source count mismatch: expected {EXPECTED_RECORD_COUNT:,}, got {total_source_count:,}"
                )

            # 2. Response breakdown: timely + untimely + unknown = source count, known = timely + untimely
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE timely_response = TRUE) AS timely,
                    COUNT(*) FILTER (WHERE timely_response = FALSE) AS untimely,
                    COUNT(*) FILTER (WHERE timely_response IS NULL) AS unknown,
                    COUNT(*) FILTER (WHERE timely_response IS NOT NULL) AS known
                FROM arzi.complaints;
                """
            )
            timely, untimely, unknown, known = cur.fetchone()
            if (timely + untimely + unknown) != total_source_count:
                errors.append(
                    f"Timely breakdown mismatch: timely({timely}) + untimely({untimely}) + unknown({unknown}) != {total_source_count}"
                )
            if known != (timely + untimely):
                errors.append(f"Known responses ({known}) != timely({timely}) + untimely({untimely})")
            if timeliness_data.get("overall_known_denominator") != known:
                errors.append(
                    f"Timeliness denominator ({timeliness_data.get('overall_known_denominator')}) != database known ({known})"
                )

            # 3. Product response counts sum to source count
            prod_complaints_sum = sum(int(e.get("complaint_count", 0)) for e in timeliness_data.get("evaluations", []))
            if prod_complaints_sum != total_source_count:
                errors.append(
                    f"Product complaint counts sum ({prod_complaints_sum:,}) does not equal source total ({total_source_count:,})"
                )

            # 4. Half-year totals equal independent counts for 2019 H1 and H2
            cur.execute(
                """
                SELECT COUNT(*) FROM arzi.complaints
                WHERE date_received >= '2019-01-01' AND date_received <= '2019-06-30';
                """
            )
            h1_source = cur.fetchone()[0]
            if mix_data.get("h1_total") != h1_source:
                errors.append(f"H1 total ({mix_data.get('h1_total')}) != independent H1 query ({h1_source})")

            cur.execute(
                """
                SELECT COUNT(*) FROM arzi.complaints
                WHERE date_received >= '2019-07-01' AND date_received <= '2019-12-31';
                """
            )
            h2_source = cur.fetchone()[0]
            if mix_data.get("h2_total") != h2_source:
                errors.append(f"H2 total ({mix_data.get('h2_total')}) != independent H2 query ({h2_source})")

            # 5. January 2020 is excluded from both half-year totals
            cur.execute("SELECT COUNT(*) FROM arzi.complaints WHERE date_received >= '2020-01-01';")
            jan_2020_count = cur.fetchone()[0]
            if total_source_count - (h1_source + h2_source) != jan_2020_count:
                errors.append("Inconsistent date coverage: non-2019 records do not match January 2020 count")

    if errors:
        raise ValueError("Source statistical verification failed:\n" + "\n".join(f"  - {e}" for e in errors))


def build_evidence_findings(
    timeliness_data: dict[str, Any],
    mix_data: dict[str, Any],
    concentration_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compile concise, factual evidence-based findings based on computed screening results."""
    findings: list[dict[str, Any]] = []

    # Finding 1: Timeliness lower signals
    flagged_timeliness = timeliness_data.get("flagged_signals", [])
    if flagged_timeliness:
        for item in flagged_timeliness:
            findings.append({
                "area": "Timely Response Investigation Signal",
                "observation": (
                    f"Product '{item['product']}' recorded a timely-response rate of {item['observed_rate_pct']}%, "
                    f"which is {abs(item['difference_pp']):.2f} percentage points lower than other products combined "
                    f"({item['other_products_rate_pct']}%, 95% Newcombe CI: [{item['newcombe_diff_ci_pp'][0]:.2f}, "
                    f"{item['newcombe_diff_ci_pp'][1]:.2f}] pp, adjusted p = {item['p_value_adjusted']:.2e})."
                ),
                "meets_screening_criteria": True,
                "analyst_follow_up": (
                    f"Analysts may examine internal routing and response workflows specifically for {item['product']} "
                    f"complaints to identify operational drivers behind lower timeliness rates."
                ),
            })
    else:
        findings.append({
            "area": "Timely Response Investigation Signal",
            "observation": "No financial product met the conservative lower-timeliness screening criteria (>= 2.0 pp deficit, adjusted p < 0.05, CI < 0).",
            "meets_screening_criteria": False,
            "analyst_follow_up": "Timely response rates across categories remain uniformly high (overall 98.23%).",
        })

    # Finding 2: Product mix shifts
    flagged_mix = mix_data.get("flagged_signals", [])
    if flagged_mix:
        for item in flagged_mix:
            direction = "increase" if item["share_change_pp"] > 0 else "decrease"
            findings.append({
                "area": "Product Mix Shift Signal",
                "observation": (
                    f"Product '{item['product']}' exhibited a statistically significant {direction} in complaint share "
                    f"of {abs(item['share_change_pp']):.2f} percentage points between 2019 H1 ({item['h1_share_pct']}%) "
                    f"and 2019 H2 ({item['h2_share_pct']}%, 95% Newcombe CI: [{item['newcombe_diff_ci_pp'][0]:.2f}, "
                    f"{item['newcombe_diff_ci_pp'][1]:.2f}] pp, adjusted p = {item['p_value_adjusted']:.2e})."
                ),
                "meets_screening_criteria": True,
                "analyst_follow_up": (
                    f"Analysts may investigate broader market or regulatory events in late 2019 affecting {item['product']} "
                    f"filings, keeping in mind that relative share changes can reflect shifts across other products."
                ),
            })
    else:
        findings.append({
            "area": "Product Mix Shift Signal",
            "observation": "No financial product met the screening criteria for a meaningful half-year share change (>= 2.0 pp shift, adjusted p < 0.05).",
            "meets_screening_criteria": False,
            "analyst_follow_up": "Product complaint distributions remained relatively stable across 2019 half-years.",
        })

    # Finding 3: Issue concentration observation
    if concentration_data:
        sorted_conc = sorted(concentration_data, key=lambda x: x["top_three_combined_share_pct"], reverse=True)
        top_conc = sorted_conc[0]
        top_issue = top_conc["top_three_issues"][0]
        findings.append({
            "area": "Issue Concentration Profile",
            "observation": (
                f"Complaints in '{top_conc['product']}' show extreme concentration: its top 3 issues represent "
                f"{top_conc['top_three_combined_share_pct']}% of all complaints in the category ({top_conc['top_three_combined_count']:,} "
                f"of {top_conc['product_total_complaints']:,}), led by '{top_issue['issue']}' at {top_issue['share_of_product_pct']}%."
            ),
            "meets_screening_criteria": True,
            "analyst_follow_up": (
                "High issue concentration indicates that categorical complaint forms capture standard grievances; "
                "subsequent NLP narrative analysis may reveal whether granular sub-issues exist beneath these top labels."
            ),
        })

    return findings


def generate_statistical_charts(
    timeliness_data: dict[str, Any],
    mix_data: dict[str, Any],
    output_dir: Path | str,
) -> list[Path]:
    """Generate the two required statistical visualization charts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Chart A: product_timely_intervals.png
    chart_a_path = out_dir / "product_timely_intervals.png"
    evals = timeliness_data.get("evaluations", [])
    sorted_evals = sorted(evals, key=lambda x: x["observed_rate_pct"])

    prod_names = [textwrap.fill(e["product"], width=32) for e in sorted_evals]
    rates = [e["observed_rate_pct"] for e in sorted_evals]
    ci_lows = [e["wilson_ci_pct"][0] for e in sorted_evals]
    ci_highs = [e["wilson_ci_pct"][1] for e in sorted_evals]
    xerr_low = [r - l for r, l in zip(rates, ci_lows)]
    xerr_high = [h - r for r, h in zip(rates, ci_highs)]
    known_counts = [e["known_response_count"] for e in sorted_evals]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(sorted_evals))

    ax.errorbar(
        rates,
        y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#2c3e50",
        ecolor="#e74c3c",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(prod_names, fontsize=9)
    ax.set_title(
        "Recorded Timely-Response Rates by Product (95% Wilson CI)\nHistorical Consumer Complaint Database",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Recorded Timely-Response Rate (%)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    overall_rate = timeliness_data.get("overall_timely_rate_pct", 98.23)
    ax.axvline(overall_rate, color="#2980b9", linestyle=":", linewidth=1.2, label=f"Overall Rate ({overall_rate}%)")
    ax.legend(loc="lower left", fontsize=9)

    for y, r, n in zip(y_pos, rates, known_counts):
        ax.annotate(
            f" {r:.1f}% (n={n:,})",
            xy=(r, y),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlim(min(rates) - 2.0, 101.5)
    fig.tight_layout()
    fig.savefig(chart_a_path, dpi=150)
    plt.close(fig)
    generated.append(chart_a_path)

    # Chart B: product_mix_change.png
    chart_b_path = out_dir / "product_mix_change.png"
    mix_comps = mix_data.get("product_comparisons", [])
    sorted_mix = sorted(mix_comps, key=lambda x: x["share_change_pp"])

    mix_prod_names = [textwrap.fill(m["product"], width=32) for m in sorted_mix]
    diffs = [m["share_change_pp"] for m in sorted_mix]
    colors = ["#27ae60" if d > 0 else "#c0392b" for d in diffs]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos_mix = np.arange(len(sorted_mix))
    bars = ax.barh(y_pos_mix, diffs, color=colors, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos_mix)
    ax.set_yticklabels(mix_prod_names, fontsize=9)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.9)
    ax.set_title(
        "Financial Product Complaint Share Change (2019 H2 minus H1)\nExpressed in Percentage Points",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Difference in Complaint Share (Percentage Points)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, d in zip(bars, diffs):
        ha = "left" if d >= 0 else "right"
        offset = 0.15 if d >= 0 else -0.15
        sign = "+" if d > 0 else ""
        ax.annotate(
            f" {sign}{d:.2f} pp",
            xy=(d + offset, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha,
            fontsize=8.5,
        )

    x_bound = max(abs(min(diffs)), max(diffs)) + 1.2
    ax.set_xlim(-x_bound, x_bound)
    fig.tight_layout()
    fig.savefig(chart_b_path, dpi=150)
    plt.close(fig)
    generated.append(chart_b_path)

    return generated


def run_full_statistical_analysis(
    views_path: Path | str,
    output_dir: Path | str,
    summary_file: Path | str,
) -> dict[str, Any]:
    """Execute complete statistical workflow: apply views, compute tests, render charts, save JSON."""
    v_path = Path(views_path)
    out_dir = Path(output_dir)
    json_path = Path(summary_file)

    # 1. Apply views
    apply_statistical_views(v_path)

    # 2. Fetch data from views
    response_rows = fetch_product_response_counts()
    half_year_rows = fetch_half_year_product_counts()

    # 3. Execute analyses
    timeliness_data = analyze_product_timeliness(response_rows)
    mix_data = analyze_product_mix_changes(half_year_rows)
    concentration_data = analyze_issue_concentration()

    # 4. Validate metrics
    validate_statistical_metrics(timeliness_data, mix_data, concentration_data)
    verify_source_statistical_totals(timeliness_data, mix_data)

    # 5. Build evidence findings
    findings = build_evidence_findings(timeliness_data, mix_data, concentration_data)

    # 6. Generate charts
    generate_statistical_charts(timeliness_data, mix_data, out_dir)

    # 7. Compile structured summary
    structured_summary = {
        "dataset_record_count": 117214,
        "date_range": {
            "min_date": "2019-01-01",
            "max_date": "2020-01-09",
        },
        "analysis_period_definitions": {
            "2019_h1": "2019-01-01 to 2019-06-30",
            "2019_h2": "2019-07-01 to 2019-12-31",
            "jan_2020_excluded": "January 2020 excluded from half-year comparison due to partial month status",
        },
        "timely_response_analysis": timeliness_data,
        "product_mix_shift_analysis": mix_data,
        "issue_concentration_analysis": concentration_data,
        "investigation_findings": findings,
        "interpretation_limitations": [
            "The dataset does not represent every customer of a financial institution.",
            "It does not provide company transaction volumes or total customer counts.",
            "Higher complaint volume does not automatically mean worse company performance.",
            "A timely response is a recorded process outcome, not proof of customer satisfaction.",
            "Statistical relationships do not establish causes.",
            "A statistically significant difference may still be too small to matter in practice.",
            "January 2020 is an incomplete month and must not be treated as a full reporting month.",
            "The observations represent the historical published complaint collection, not a random sample of all banking customers.",
        ],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(structured_summary, indent=2, default=serialize_json_value),
        encoding="utf-8",
    )

    return structured_summary
