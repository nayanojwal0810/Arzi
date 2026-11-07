# Arzi — Financial Complaint Intelligence

Arzi helps analysts understand financial complaints. It combines complaint records and customer-written text to find common issues, track patterns, and support further investigation.

## Data

The project uses 117,214 historical complaints from the US Consumer Financial Protection Bureau (CFPB), covering January 2019 to January 2020. January 2020 contains only nine days of records.

Download the [historical complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) and save it as `data/raw/complaints.csv.gz`. Keep the file compressed. The dataset stays on your computer and is not committed to GitHub.

## Getting Started

Use Python 3.11 or newer and a running PostgreSQL server. Configure local PostgreSQL authentication before running:

```bash
python -m pip install -e ".[dev]"
python scripts/load_data.py
python scripts/run_analysis.py
python -m pytest -q
```

The loader creates the `arzi` database if needed, preserves the source records in `arzi.complaints_raw`, and stores prepared records in `arzi.complaints`.

## Current Analysis

PostgreSQL views calculate complaint totals, product and issue distributions, company volumes, response patterns, submission channels, and monthly trends. Python creates three EDA charts and a local JSON summary under `data/processed/`.

The dataset includes nine products, 79 distinct issues, and 2,541 companies. Credit reporting is the largest product category, with 55,558 complaints (47.40%). These counts describe published complaint records, not the quality of a company or the full customer population.

## Project Files

- `src/arzi/` — data preparation, database access, and analysis.
- `scripts/` — commands to load and analyze complaints.
- `sql/` — table definitions, analysis views, and checks.
- `tests/` — basic checks for the Python code.
