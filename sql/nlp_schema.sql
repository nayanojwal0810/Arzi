-- =============================================================================
-- Arzi NLP Topic Discovery Schema and Analytical Views
-- =============================================================================

CREATE TABLE IF NOT EXISTS arzi.topic_catalog (
    topic_id INTEGER PRIMARY KEY,
    top_terms TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS arzi.complaint_topics (
    complaint_id BIGINT PRIMARY KEY,
    topic_id INTEGER NULL,
    topic_strength DOUBLE PRECISION NULL,
    usable_for_nlp BOOLEAN NOT NULL,
    CONSTRAINT fk_topic_catalog FOREIGN KEY (topic_id) REFERENCES arzi.topic_catalog(topic_id),
    CONSTRAINT fk_complaint FOREIGN KEY (complaint_id) REFERENCES arzi.complaints(complaint_id)
);

CREATE OR REPLACE VIEW arzi.v_topic_summary AS
SELECT
    ct.topic_id,
    COUNT(*) AS complaint_count,
    ROUND(
        (COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM arzi.complaint_topics), 0)),
        2
    ) AS complaint_share_pct
FROM arzi.complaint_topics ct
WHERE ct.topic_id IS NOT NULL
GROUP BY ct.topic_id
ORDER BY ct.topic_id ASC;

CREATE OR REPLACE VIEW arzi.v_topic_product_summary AS
WITH product_totals AS (
    SELECT product, COUNT(*) AS product_total_complaints
    FROM arzi.complaints
    GROUP BY product
),
topic_prods AS (
    SELECT t.topic_id, p.product
    FROM arzi.topic_catalog t
    CROSS JOIN (SELECT DISTINCT product FROM arzi.complaints) p
)
SELECT
    tp.topic_id,
    tp.product,
    COUNT(ct.complaint_id) AS complaint_count,
    pt.product_total_complaints,
    ROUND(
        (COUNT(ct.complaint_id) * 100.0 / NULLIF(pt.product_total_complaints, 0)),
        2
    ) AS within_product_share_pct
FROM topic_prods tp
JOIN product_totals pt ON tp.product = pt.product
LEFT JOIN (
    arzi.complaint_topics ct
    JOIN arzi.complaints c ON ct.complaint_id = c.complaint_id
) ON ct.topic_id = tp.topic_id AND c.product = tp.product
GROUP BY tp.topic_id, tp.product, pt.product_total_complaints
ORDER BY tp.topic_id ASC, tp.product ASC;

CREATE OR REPLACE VIEW arzi.v_topic_issue_summary AS
SELECT
    ct.topic_id,
    c.issue,
    COUNT(*) AS complaint_count
FROM arzi.complaint_topics ct
JOIN arzi.complaints c ON ct.complaint_id = c.complaint_id
WHERE ct.topic_id IS NOT NULL
GROUP BY ct.topic_id, c.issue
ORDER BY ct.topic_id ASC, complaint_count DESC, c.issue ASC;
