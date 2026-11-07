-- Validation checks for Arzi data pipeline

-- 1. Total raw records
SELECT COUNT(*) AS total_raw_records
FROM arzi.complaints_raw;

-- 2. Total prepared records
SELECT COUNT(*) AS total_prepared_records
FROM arzi.complaints;

-- 3. Distinct complaint IDs
SELECT COUNT(DISTINCT complaint_id) AS distinct_complaint_ids
FROM arzi.complaints;

-- 4. Missing complaint IDs
SELECT COUNT(*) AS missing_complaint_ids
FROM arzi.complaints
WHERE complaint_id IS NULL;

-- 5. Duplicate complaint IDs
SELECT complaint_id, COUNT(*) AS id_count
FROM arzi.complaints
GROUP BY complaint_id
HAVING COUNT(*) > 1;

-- 6. Date range (minimum and maximum complaint dates)
SELECT
    MIN(date_received) AS min_date_received,
    MAX(date_received) AS max_date_received
FROM arzi.complaints;

-- 7. Number of records with usable narratives
SELECT COUNT(*) AS records_with_narratives
FROM arzi.complaints
WHERE has_narrative = TRUE;

-- 8. Number of repeated narrative texts
SELECT COALESCE(SUM(dup_count), 0) AS repeated_narratives_count
FROM (
    SELECT COUNT(*) - 1 AS dup_count
    FROM arzi.complaints
    WHERE has_narrative = TRUE
    GROUP BY consumer_complaint_narrative
    HAVING COUNT(*) > 1
) sub;

-- 9. Distinct products, issues, and companies
SELECT
    COUNT(DISTINCT product) AS distinct_products,
    COUNT(DISTINCT issue) AS distinct_issues,
    COUNT(DISTINCT company) AS distinct_companies
FROM arzi.complaints;

-- 10. Values and distribution of timely response
SELECT
    timely_response,
    COUNT(*) AS record_count
FROM arzi.complaints
GROUP BY timely_response
ORDER BY record_count DESC;
