-- Statistical views for Arzi — Financial Complaint Intelligence

-- View A: Product Timely-Response Counts
CREATE OR REPLACE VIEW arzi.v_product_response_counts AS
SELECT
    product,
    COUNT(*) AS complaint_count,
    COUNT(*) FILTER (WHERE timely_response IS NOT NULL) AS known_response_count,
    COUNT(*) FILTER (WHERE timely_response = TRUE) AS timely_count,
    COUNT(*) FILTER (WHERE timely_response = FALSE) AS untimely_count,
    COUNT(*) FILTER (WHERE timely_response IS NULL) AS unknown_response_count
FROM arzi.complaints
GROUP BY product
ORDER BY complaint_count DESC, product ASC;

-- View B: Half-Year Product Counts (2019 H1 vs H2)
CREATE OR REPLACE VIEW arzi.v_half_year_product_counts AS
WITH periods AS (
    SELECT '2019_h1' AS period, '2019-01-01'::DATE AS start_date, '2019-06-30'::DATE AS end_date
    UNION ALL
    SELECT '2019_h2' AS period, '2019-07-01'::DATE AS start_date, '2019-12-31'::DATE AS end_date
),
all_products AS (
    SELECT DISTINCT product FROM arzi.complaints
),
grid AS (
    SELECT p.period, p.start_date, p.end_date, prod.product
    FROM periods p
    CROSS JOIN all_products prod
),
counts AS (
    SELECT
        g.period,
        g.product,
        COUNT(c.complaint_id) AS complaint_count
    FROM grid g
    LEFT JOIN arzi.complaints c
        ON c.product = g.product
        AND c.date_received >= g.start_date
        AND c.date_received <= g.end_date
    GROUP BY g.period, g.product
)
SELECT
    c.period,
    c.product,
    c.complaint_count,
    SUM(c.complaint_count) OVER (PARTITION BY c.period) AS period_total
FROM counts c
ORDER BY c.period ASC, c.complaint_count DESC, c.product ASC;
