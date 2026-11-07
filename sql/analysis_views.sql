-- Analytical views for Arzi — Financial Complaint Intelligence

-- View 1: Product Summary
CREATE OR REPLACE VIEW arzi.v_product_summary AS
SELECT
    product,
    COUNT(*) AS complaint_count,
    ROUND(
        (COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0)),
        2
    ) AS complaint_share_pct,
    COUNT(DISTINCT issue) AS distinct_issue_count,
    COUNT(DISTINCT company) AS distinct_company_count,
    COUNT(*) FILTER (WHERE has_narrative = TRUE) AS narrative_count,
    COUNT(*) FILTER (WHERE timely_response IS NOT NULL) AS known_timely_response_count,
    ROUND(
        (COUNT(*) FILTER (WHERE timely_response = TRUE) * 100.0 / NULLIF(COUNT(*) FILTER (WHERE timely_response IS NOT NULL), 0)),
        2
    ) AS timely_response_rate_pct
FROM arzi.complaints
GROUP BY product
ORDER BY complaint_count DESC;

-- View 2: Monthly Trend
CREATE OR REPLACE VIEW arzi.v_monthly_trend AS
WITH date_bounds AS (
    SELECT
        DATE_TRUNC('month', MIN(date_received))::DATE AS min_month,
        DATE_TRUNC('month', MAX(date_received))::DATE AS max_month,
        MAX(date_received) AS max_date
    FROM arzi.complaints
),
month_series AS (
    SELECT generate_series(
        (SELECT min_month FROM date_bounds),
        (SELECT max_month FROM date_bounds),
        INTERVAL '1 month'
    )::DATE AS month_start
),
monthly_counts AS (
    SELECT
        ms.month_start,
        COUNT(c.complaint_id) AS complaint_count
    FROM month_series ms
    LEFT JOIN arzi.complaints c
        ON DATE_TRUNC('month', c.date_received)::DATE = ms.month_start
    GROUP BY ms.month_start
),
lagged AS (
    SELECT
        mc.month_start,
        mc.complaint_count,
        LAG(mc.complaint_count) OVER (ORDER BY mc.month_start) AS previous_month_count,
        mc.complaint_count - LAG(mc.complaint_count) OVER (ORDER BY mc.month_start) AS month_over_month_change,
        CASE
            WHEN mc.month_start = (SELECT max_month FROM date_bounds)
                 AND (SELECT max_date FROM date_bounds) < (SELECT (max_month + INTERVAL '1 month - 1 day')::DATE FROM date_bounds)
            THEN TRUE
            ELSE FALSE
        END AS is_partial_last_month
    FROM monthly_counts mc
)
SELECT
    month_start,
    complaint_count,
    previous_month_count,
    month_over_month_change,
    CASE
        WHEN is_partial_last_month THEN NULL
        WHEN previous_month_count IS NULL OR previous_month_count <= 0 THEN NULL
        ELSE ROUND((month_over_month_change * 100.0 / previous_month_count), 2)
    END AS month_over_month_change_pct,
    is_partial_last_month
FROM lagged
ORDER BY month_start;

-- View 3: Monthly Product Trend
CREATE OR REPLACE VIEW arzi.v_monthly_product_trend AS
SELECT
    DATE_TRUNC('month', date_received)::DATE AS month_start,
    product,
    COUNT(*) AS complaint_count
FROM arzi.complaints
GROUP BY DATE_TRUNC('month', date_received)::DATE, product
ORDER BY month_start, product;

-- View 4: Issue Summary
CREATE OR REPLACE VIEW arzi.v_issue_summary AS
SELECT
    product,
    issue,
    COUNT(*) AS complaint_count,
    ROUND(
        (COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY product), 0)),
        2
    ) AS product_complaint_share_pct
FROM arzi.complaints
GROUP BY product, issue
ORDER BY product, complaint_count DESC;

-- View 5: Company Summary
CREATE OR REPLACE VIEW arzi.v_company_summary AS
SELECT
    company,
    COUNT(*) AS complaint_count,
    COUNT(DISTINCT product) AS distinct_product_count,
    COUNT(*) FILTER (WHERE timely_response IS NOT NULL) AS known_timely_response_count,
    ROUND(
        (COUNT(*) FILTER (WHERE timely_response = TRUE) * 100.0 / NULLIF(COUNT(*) FILTER (WHERE timely_response IS NOT NULL), 0)),
        2
    ) AS timely_response_rate_pct
FROM arzi.complaints
GROUP BY company
ORDER BY complaint_count DESC;

-- View 6: Response Summary
CREATE OR REPLACE VIEW arzi.v_response_summary AS
SELECT
    product,
    COALESCE(company_response_to_consumer, 'Unknown') AS company_response_to_consumer,
    COUNT(*) AS complaint_count,
    ROUND(
        (COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY product), 0)),
        2
    ) AS product_response_share_pct
FROM arzi.complaints
GROUP BY product, company_response_to_consumer
ORDER BY product, complaint_count DESC;

-- View 7: Channel Summary
CREATE OR REPLACE VIEW arzi.v_channel_summary AS
SELECT
    COALESCE(submitted_via, 'Unknown') AS submitted_via,
    COUNT(*) AS complaint_count,
    ROUND(
        (COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0)),
        2
    ) AS complaint_share_pct
FROM arzi.complaints
GROUP BY COALESCE(submitted_via, 'Unknown')
ORDER BY complaint_count DESC;
