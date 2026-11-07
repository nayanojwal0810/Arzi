# Arzi — Financial Complaint Intelligence

Arzi helps analysts understand financial complaints. It combines complaint records and customer-written text to find common issues, compare patterns, and support further investigation.

## Data

The project uses 117,214 historical complaints from the US Consumer Financial Protection Bureau (CFPB), covering January 2019 to January 2020. January 2020 includes only nine days of records.

Download the [historical complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) and save it as `data/raw/complaints.csv.gz`. Keep the compressed file unchanged. The raw data stays on your computer.

## Getting Started

Use Python 3.11 or newer and a running PostgreSQL server. Configure local PostgreSQL authentication, then run:

```bash
python -m pip install -e ".[dev]"
python scripts/load_data.py
python scripts/run_analysis.py
python scripts/run_statistics.py
python -m pytest -q
```

The loader creates the `arzi` database if needed and stores the original and prepared complaints separately. Analysis scripts create local charts and JSON results under `data/processed/`.

## Findings

- Credit reporting accounts for 55,558 complaints (47.40%) in this dataset.
- Overall, 115,136 of 117,214 complaints (98.23%) have a recorded timely response. Payday, title, or personal loans and debt collection show lower recorded rates than the other products combined.
- Credit reporting's share of complaints increased from 45.34% in the first half of 2019 to 49.33% in the second half.

These findings describe published historical complaints. They do not measure overall customer satisfaction, prove wrongdoing, or establish why patterns changed.

## Project Files

- `src/arzi/` — data preparation, database access, SQL analysis, and statistics.
- `scripts/` — commands to load and analyze complaints.
- `sql/` — PostgreSQL tables, analysis views, and checks.
- `tests/` — basic checks for the Python code.
