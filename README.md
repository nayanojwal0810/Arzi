# Arzi — Financial Complaint Intelligence

Arzi helps analysts explore customer complaints about financial services. It combines complaint details and customer-written text to study common issues, compare patterns, and support further investigation.

## Data

The project uses 117,214 historical complaints from the US Consumer Financial Protection Bureau (CFPB), covering January 2019 to January 2020.

Download the [complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) and save it as `data/raw/complaints.csv.gz`. Keep the compressed file unchanged. The raw dataset stays on your computer and is not included in GitHub.

## Getting Started

Use Python 3.11 or newer and a running PostgreSQL server. Configure your local PostgreSQL login before loading the data.

From the project folder, run:

```bash
python -m pip install -e ".[dev]"
python scripts/load_data.py
python -m pytest -q
```

The loader checks the source file, creates the `arzi` database if needed, and loads 117,214 records into `arzi.complaints_raw` and `arzi.complaints`. It checks the loaded data before saving changes. Repeated complaint narratives are retained.

## Project Files

- `src/arzi/` — data preparation and database connection code.
- `scripts/load_data.py` — runs the data loading process.
- `sql/schema.sql` — defines the PostgreSQL tables and indexes.
- `sql/checks.sql` — contains data verification queries.
- `tests/` — basic checks for data preparation.
