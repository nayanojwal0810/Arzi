# Data Profile

## Source
- **Dataset Name:** CFPB Consumer Complaint Database (Historical Snapshot)
- **Source URL:** https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz
- **Snapshot Reference:** smltar historical complaint archive (commit 3ae4ab3)
- **Expected SHA-256:** `f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b`
- **Actual SHA-256:** `f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b`
- **Verification Status:** VERIFIED (Checksum match and schema confirmed)

## Dataset Overview
- **Record Count:** 117,214 rows
- **Column Count:** 18 columns
- **Date Range:** 2019-01-01 to 2020-01-09
- **Main Fields:** `complaint_id`, `date_received`, `product`, `sub_product`, `issue`, `sub_issue`, `consumer_complaint_narrative`, `company`, `state`, `zip_code`, `company_response_to_consumer`, `timely_response`
- **Distinct Products:** 9
- **Distinct Issues:** 79
- **Distinct Companies:** 2,541

## Data Quality
- **Duplicate Full Rows:** 0
- **Complaint Identifier Integrity:** 0 missing, 0 duplicate IDs (117,214 unique IDs verified)
- **Narrative Availability:** 117,214 usable narratives (100.0%), 0 unusable (missing, whitespace, or placeholder); 10,854 exact duplicate narrative texts (common boilerplate or standardized submissions)
- **Missing Values by Column:**
  - `consumer_disputed`: 117,214 missing (100.0%)
  - `tags`: 97,503 missing (83.2%)
  - `company_public_response`: 56,156 missing (47.9%)
  - `zip_code`: 26,196 missing (22.3%)
  - `sub_issue`: 15,764 missing (13.4%)
  - `state`: 514 missing (0.4%)
  - `date_received`: 0 missing (0.0%)
  - `product`: 0 missing (0.0%)
  - `sub_product`: 0 missing (0.0%)
  - `issue`: 0 missing (0.0%)
  - `consumer_complaint_narrative`: 0 missing (0.0%)
  - `company`: 0 missing (0.0%)
  - `consumer_consent_provided`: 0 missing (0.0%)
  - `submitted_via`: 0 missing (0.0%)
  - `date_sent_to_company`: 0 missing (0.0%)
  - `company_response_to_consumer`: 0 missing (0.0%)
  - `timely_response`: 0 missing (0.0%)
  - `complaint_id`: 0 missing (0.0%)

## Limitations
- **Temporal Window:** The dataset spans from 2019-01-01 to 2020-01-09, serving as a historical baseline snapshot rather than a continuously updated real-time stream.
- **Narrative Selection:** 100% of the records in this historical snapshot contain consumer complaint narratives, representing a cohort of complaints with public written text rather than the wider set of non-narrative filings.
- **Unpopulated Attributes:** The `consumer_disputed` column contains no populated values across all 117,214 records, and optional metadata such as `tags` (97,503 missing) and `company_public_response` (56,156 missing) exhibit high missingness.
