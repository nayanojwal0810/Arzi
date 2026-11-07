-- Schema and table definitions for Arzi — Financial Complaint Intelligence

CREATE SCHEMA IF NOT EXISTS arzi;

-- Raw complaints table: stores original source fields exactly as received
CREATE TABLE IF NOT EXISTS arzi.complaints_raw (
    date_received TEXT,
    product TEXT,
    sub_product TEXT,
    issue TEXT,
    sub_issue TEXT,
    consumer_complaint_narrative TEXT,
    company_public_response TEXT,
    company TEXT,
    state TEXT,
    zip_code TEXT,
    tags TEXT,
    consumer_consent_provided TEXT,
    submitted_via TEXT,
    date_sent_to_company TEXT,
    company_response_to_consumer TEXT,
    timely_response TEXT,
    consumer_disputed TEXT,
    complaint_id TEXT
);

-- Prepared complaints table: standardized types and constraints for analysis
CREATE TABLE IF NOT EXISTS arzi.complaints (
    complaint_id BIGINT PRIMARY KEY,
    date_received DATE NOT NULL,
    date_sent_to_company DATE,
    product TEXT NOT NULL,
    sub_product TEXT,
    issue TEXT NOT NULL,
    sub_issue TEXT,
    consumer_complaint_narrative TEXT,
    company_public_response TEXT,
    company TEXT NOT NULL,
    state TEXT,
    zip_code TEXT,
    tags TEXT,
    consumer_consent_provided TEXT,
    submitted_via TEXT,
    company_response_to_consumer TEXT,
    timely_response BOOLEAN,
    consumer_disputed TEXT,
    has_narrative BOOLEAN NOT NULL
);

-- Indexes for analytical queries
CREATE INDEX IF NOT EXISTS idx_complaints_date_received ON arzi.complaints (date_received);
CREATE INDEX IF NOT EXISTS idx_complaints_product ON arzi.complaints (product);
CREATE INDEX IF NOT EXISTS idx_complaints_issue ON arzi.complaints (issue);
CREATE INDEX IF NOT EXISTS idx_complaints_company ON arzi.complaints (company);
