"""Unit tests for Arzi statistical analysis routines using synthetic data."""

from __future__ import annotations

import pytest

from arzi.statistics import (
    analyze_issue_concentration,
    analyze_product_mix_changes,
    analyze_product_timeliness,
    apply_benjamini_hochberg,
    calculate_newcombe_diff_ci,
    calculate_two_proportion_pvalue,
    calculate_wilson_ci,
    validate_statistical_metrics,
)


def test_wilson_confidence_interval_bounds() -> None:
    ci = calculate_wilson_ci(95, 100)
    assert ci is not None
    low, high = ci
    assert 0.0 <= low <= high <= 1.0
    assert low < 0.95 < high


def test_wilson_zero_denominator_handling() -> None:
    assert calculate_wilson_ci(0, 0) is None
    assert calculate_wilson_ci(10, 0) is None
    assert calculate_wilson_ci(-1, 50) is None
    assert calculate_wilson_ci(60, 50) is None


def test_newcombe_diff_ci_and_zero_denominator() -> None:
    ci = calculate_newcombe_diff_ci(90, 100, 80, 100)
    assert ci is not None
    low, high = ci
    assert low <= 0.10 <= high

    # Zero denominator
    assert calculate_newcombe_diff_ci(10, 0, 5, 20) is None
    assert calculate_newcombe_diff_ci(5, 20, 10, 0) is None


def test_two_proportion_pvalue() -> None:
    pval = calculate_two_proportion_pvalue(50, 100, 50, 100)
    assert pval is not None
    assert pytest.approx(pval, abs=1e-3) == 1.0

    pval_diff = calculate_two_proportion_pvalue(95, 100, 50, 100)
    assert pval_diff is not None
    assert pval_diff < 0.001

    assert calculate_two_proportion_pvalue(50, 0, 50, 100) is None


def test_benjamini_hochberg_adjustment() -> None:
    raw_pvals = [0.01, 0.04, 0.08]
    adjusted = apply_benjamini_hochberg(raw_pvals)

    # For [0.01, 0.04, 0.08]:
    # rank 1: 0.01 * 3 / 1 = 0.03
    # rank 2: 0.04 * 3 / 2 = 0.06
    # rank 3: 0.08 * 3 / 3 = 0.08
    assert len(adjusted) == 3
    assert pytest.approx(adjusted[0], abs=1e-4) == 0.03
    assert pytest.approx(adjusted[1], abs=1e-4) == 0.06
    assert pytest.approx(adjusted[2], abs=1e-4) == 0.08

    # Empty list
    assert apply_benjamini_hochberg([]) == []


def test_product_timeliness_screening_and_sample_size() -> None:
    # Product A: large sample, deficit >= 2 pp, significant
    # Product B: small sample (n=50 < 200), deficit >= 2 pp -> must NOT meet screening
    # Product C: large sample, but deficit only 0.5 pp -> must NOT meet screening
    synthetic_rows = [
        {
            "product": "Product A",
            "complaint_count": 1000,
            "known_response_count": 1000,
            "timely_count": 900,  # 90.0%
            "untimely_count": 100,
            "unknown_response_count": 0,
        },
        {
            "product": "Product B",
            "complaint_count": 50,
            "known_response_count": 50,
            "timely_count": 40,  # 80.0%
            "untimely_count": 10,
            "unknown_response_count": 0,
        },
        {
            "product": "Product C",
            "complaint_count": 5000,
            "known_response_count": 5000,
            "timely_count": 4900,  # 98.0%
            "untimely_count": 100,
            "unknown_response_count": 0,
        },
    ]

    analysis = analyze_product_timeliness(synthetic_rows)
    evals = {e["product"]: e for e in analysis["evaluations"]}

    assert evals["Product A"]["is_sample_size_eligible"] is True
    # Product A is lower than others by > 2 pp and should meet screening
    assert evals["Product A"]["meets_screening_criteria"] is True

    # Product B has n=50 (< 200) -> disqualified by sample size
    assert evals["Product B"]["is_sample_size_eligible"] is False
    assert evals["Product B"]["meets_screening_criteria"] is False

    # Product C has high timeliness (98%) -> not a lower-timeliness signal
    assert evals["Product C"]["meets_screening_criteria"] is False


def test_product_mix_screening_rules() -> None:
    # 2 products, H1 total = 1000, H2 total = 1000
    # Product 1: H1 = 400 (40%), H2 = 500 (50%) -> shift = +10.0 pp (meets criteria)
    # Product 2: H1 = 600 (60%), H2 = 500 (50%) -> shift = -10.0 pp (meets criteria)
    synthetic_half_year = [
        {"period": "2019_h1", "product": "P1", "complaint_count": 400, "period_total": 1000},
        {"period": "2019_h1", "product": "P2", "complaint_count": 600, "period_total": 1000},
        {"period": "2019_h2", "product": "P1", "complaint_count": 500, "period_total": 1000},
        {"period": "2019_h2", "product": "P2", "complaint_count": 500, "period_total": 1000},
    ]

    mix = analyze_product_mix_changes(synthetic_half_year)
    comps = {c["product"]: c for c in mix["product_comparisons"]}

    assert comps["P1"]["share_change_pp"] == 10.0
    assert comps["P1"]["meets_screening_criteria"] is True
    assert comps["P2"]["share_change_pp"] == -10.0
    assert comps["P2"]["meets_screening_criteria"] is True
    assert mix["chi_square"]["degrees_of_freedom"] == 1


def test_product_versus_rest_difference_calculation() -> None:
    synthetic_rows = [
        {
            "product": "Product X",
            "complaint_count": 500,
            "known_response_count": 500,
            "timely_count": 450,  # 90.0%
            "untimely_count": 50,
            "unknown_response_count": 0,
        },
        {
            "product": "Product Y",
            "complaint_count": 1000,
            "known_response_count": 1000,
            "timely_count": 980,  # 98.0%
            "untimely_count": 20,
            "unknown_response_count": 0,
        },
        {
            "product": "Product Z",
            "complaint_count": 500,
            "known_response_count": 500,
            "timely_count": 490,  # 98.0%
            "untimely_count": 10,
            "unknown_response_count": 0,
        },
    ]

    analysis = analyze_product_timeliness(synthetic_rows)
    evals = {e["product"]: e for e in analysis["evaluations"]}

    # For Product X: 450/500 = 90.0%
    # Rest: (980 + 490) / (1000 + 500) = 1470 / 1500 = 98.0%
    # Difference: 90.0% - 98.0% = -8.00 pp
    px = evals["Product X"]
    assert px["observed_rate_pct"] == 90.0
    assert px["other_products_rate_pct"] == 98.0
    assert px["difference_pp"] == -8.00
    ci_low, ci_high = px["newcombe_diff_ci_pp"]
    assert ci_low is not None and ci_high is not None
    assert ci_low <= -8.00 <= ci_high
    assert ci_high < 0.0


def test_percentage_point_interpretation() -> None:
    # 90% vs 98% is a difference of -8.0 percentage points, not a -8.16% relative change
    p1 = 90.0
    p2 = 98.0
    diff_pp = round(p1 - p2, 2)
    assert diff_pp == -8.00


def test_half_year_complaint_share_calculation() -> None:
    synthetic_half_year = [
        {"period": "2019_h1", "product": "P1", "complaint_count": 300, "period_total": 1000},
        {"period": "2019_h1", "product": "P2", "complaint_count": 700, "period_total": 1000},
        {"period": "2019_h2", "product": "P1", "complaint_count": 400, "period_total": 1000},
        {"period": "2019_h2", "product": "P2", "complaint_count": 600, "period_total": 1000},
    ]
    mix = analyze_product_mix_changes(synthetic_half_year)
    comps = {c["product"]: c for c in mix["product_comparisons"]}

    assert comps["P1"]["h1_share_pct"] == 30.0
    assert comps["P1"]["h2_share_pct"] == 40.0
    assert comps["P1"]["share_change_pp"] == 10.0

    assert comps["P2"]["h1_share_pct"] == 70.0
    assert comps["P2"]["h2_share_pct"] == 60.0
    assert comps["P2"]["share_change_pp"] == -10.0


def test_top_three_issue_concentration() -> None:
    synthetic_issues = [
        ("Mortgage", "Issue A", 500),
        ("Mortgage", "Issue B", 300),
        ("Mortgage", "Issue C", 150),
        ("Mortgage", "Issue D", 50),
    ]
    synthetic_prods = [
        ("Mortgage", 1000),
    ]

    conc = analyze_issue_concentration(issue_rows=synthetic_issues, prod_rows=synthetic_prods)
    assert len(conc) == 1
    m = conc[0]
    assert m["product"] == "Mortgage"
    assert m["product_total_complaints"] == 1000
    assert len(m["top_three_issues"]) == 3
    assert m["top_three_issues"][0]["issue"] == "Issue A"
    assert m["top_three_issues"][0]["share_of_product_pct"] == 50.0
    assert m["top_three_issues"][1]["issue"] == "Issue B"
    assert m["top_three_issues"][1]["share_of_product_pct"] == 30.0
    assert m["top_three_issues"][2]["issue"] == "Issue C"
    assert m["top_three_issues"][2]["share_of_product_pct"] == 15.0
    assert m["top_three_combined_count"] == 950
    assert m["top_three_combined_share_pct"] == 95.0


def test_validate_statistical_metrics_catches_invalid_bounds() -> None:
    timeliness_data = {
        "overall_known_denominator": 100,
        "evaluations": [
            {
                "product": "TestProd",
                "known_response_count": 100,
                "wilson_ci_pct": (105.0, 95.0),  # Invalid bounds (low > high)
                "p_value_adjusted": 0.05,
            }
        ],
    }
    mix_data = {
        "h1_total": 90,
        "h2_total": 90,
        "product_comparisons": [{"h1_count": 10, "h2_count": 10}] * 9,
    }
    concentration_data = []

    with pytest.raises(ValueError, match="Invalid Wilson CI bounds"):
        validate_statistical_metrics(timeliness_data, mix_data, concentration_data)


def test_unrounded_screening_boundary_timeliness() -> None:
    # Baseline: 100,000 / 100,000 = 100.0% timely
    # Product A: 9801 / 10000 = 98.01% -> diff = -1.99 pp (fails threshold)
    # Product B: 9790 / 10000 = 97.90% -> diff = -2.10 pp (meets threshold)
    rows_a = [
        {
            "product": "Product A",
            "complaint_count": 10000,
            "known_response_count": 10000,
            "timely_count": 9801,
            "untimely_count": 199,
            "unknown_response_count": 0,
        },
        {
            "product": "Baseline",
            "complaint_count": 100000,
            "known_response_count": 100000,
            "timely_count": 100000,
            "untimely_count": 0,
            "unknown_response_count": 0,
        },
    ]

    evals_a = {e["product"]: e for e in analyze_product_timeliness(rows_a)["evaluations"]}
    assert evals_a["Product A"]["meets_screening_criteria"] is False

    rows_b = [
        {
            "product": "Product B",
            "complaint_count": 10000,
            "known_response_count": 10000,
            "timely_count": 9790,
            "untimely_count": 210,
            "unknown_response_count": 0,
        },
        {
            "product": "Baseline",
            "complaint_count": 100000,
            "known_response_count": 100000,
            "timely_count": 100000,
            "untimely_count": 0,
            "unknown_response_count": 0,
        },
    ]

    evals_b = {e["product"]: e for e in analyze_product_timeliness(rows_b)["evaluations"]}
    assert evals_b["Product B"]["meets_screening_criteria"] is True


def test_unrounded_screening_boundary_mix() -> None:
    # H1 total = 100,000, H2 total = 100,000
    # P1: H1 = 10,000 (10.0%), H2 = 11,996 (11.996%) -> unrounded diff = +1.996 pp
    # Display rounding shows +2.00 pp, but unrounded < 2.0 pp, so must NOT meet screening.
    # P2: H1 = 10,000 (10.0%), H2 = 12,050 (12.050%) -> unrounded diff = +2.050 pp, meets screening.
    synthetic_half_year = [
        {"period": "2019_h1", "product": "P1", "complaint_count": 10000, "period_total": 100000},
        {"period": "2019_h1", "product": "P2", "complaint_count": 10000, "period_total": 100000},
        {"period": "2019_h1", "product": "P3", "complaint_count": 80000, "period_total": 100000},
        {"period": "2019_h2", "product": "P1", "complaint_count": 11996, "period_total": 100000},
        {"period": "2019_h2", "product": "P2", "complaint_count": 12050, "period_total": 100000},
        {"period": "2019_h2", "product": "P3", "complaint_count": 75954, "period_total": 100000},
    ]

    mix = analyze_product_mix_changes(synthetic_half_year)
    comps = {c["product"]: c for c in mix["product_comparisons"]}

    assert comps["P1"]["share_change_pp"] == 2.0
    assert comps["P1"]["meets_screening_criteria"] is False
    assert comps["P2"]["meets_screening_criteria"] is True
