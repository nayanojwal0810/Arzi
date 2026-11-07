# Arzi

Arzi is a financial complaint intelligence project that analyzes structured complaint records and unstructured consumer narratives to support banking operations, risk identification, and regulatory compliance analysis.

## Project Overview

Financial institutions handle large volumes of consumer grievances across credit reporting, mortgages, debt collection, and banking services. Identifying root causes requires combining structured metadata with qualitative customer narratives. Arzi applies Python-based data engineering, statistical profiling, natural language processing (NLP), and analytical reporting to convert raw complaint archives into actionable operational intelligence.

## Data

This project utilizes the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database, using an established historical snapshot archive spanning 2019 to early 2020:

- **Source Archive:** [CFPB Complaints Data Archive](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz)
- **Archive SHA-256:** `f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b`

The raw dataset is not stored in GitHub due to file size considerations. Users download the compressed archive and place it in the local path `data/raw/complaints.csv.gz`.

## Setup

Set up a Python 3.11+ environment, install the package in editable mode with development dependencies, and run verification:

```bash
# Install package and dependencies
python -m pip install -e ".[dev]"

# Verify dataset integrity and generate data profile
python scripts/verify_dataset.py

# Run test suite
python -m pytest -q
```

## Repository Structure

```
arzi/
├── data/
│   └── raw/               # Local directory for raw complaint data (git-ignored)
├── reports/
│   └── data_profile.md    # Generated dataset verification report
├── scripts/
│   └── verify_dataset.py  # Command-line entry point to audit dataset integrity
├── src/
│   └── arzi/
│       ├── __init__.py    # Package initialization
│       └── data_audit.py  # Core audit routines and checksum verification
└── tests/
    └── test_data_audit.py # Unit tests using synthetic fixtures
```

## Current Functionality

The repository currently provides automated dataset auditing and validation:
- Cryptographic SHA-256 verification against the reference snapshot checksum.
- Gzip stream decompression and integrity validation.
- Schema inspection, record counts, and column missingness analysis.
- Complaint narrative usability and duplicate rate evaluation.
- Automated data profile generation in markdown format.

## Data Profile

A detailed quality audit and column breakdown is available in [reports/data_profile.md](reports/data_profile.md).
