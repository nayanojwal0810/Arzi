# Arzi — Financial Complaint Intelligence

Arzi helps analysts explore financial complaints through SQL, statistics, and customer-written text. It shows recurring problems and patterns that may deserve investigation.

## Data

The project uses 117,214 historical complaints from the US Consumer Financial Protection Bureau (CFPB), covering January 2019 to January 9, 2020. The final month is incomplete.

Download the [historical complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) and save it as `data/raw/complaints.csv.gz`. Keep the compressed file unchanged. Raw data and generated model files stay on your computer.

## Run the Project

Use Python 3.11 or newer and a running PostgreSQL server with local authentication configured.

```bash
python -m pip install -e ".[dev]"
python scripts/load_data.py
python scripts/run_analysis.py
python scripts/run_statistics.py
python scripts/run_nlp.py
python -m pytest -q
```

Use `python scripts/run_nlp.py --retrain` only when intentionally rebuilding the topic model.

## Analysis

- **SQL:** Complaint volume by product, issue, company, response, and month.
- **Statistics:** Product response-rate comparisons and changes in complaint mix between the two halves of 2019.
- **NLP:** TF-IDF and NMF discover ten provisional text topics. The pipeline retains all 117,214 complaint IDs, with 117,107 assigned a topic and 107 unassigned in the first run.

Credit reporting accounts for 55,558 complaints (47.40%). The recorded timely-response rate is 98.23%. Some NLP topics are still being checked for noisy or overly specific terms, so they should not be treated as verified complaint categories.

These are historical published complaint records. Complaint counts do not measure company quality, and the observed patterns do not prove causes.

## Project Files

- `src/arzi/` — data preparation, database access, SQL analysis, statistics, and NLP.
- `scripts/` — commands to run the project.
- `sql/` — PostgreSQL tables, views, and data checks.
- `tests/` — Python tests.

Charts, JSON summaries, and saved NLP models are written under `data/processed/` and excluded from GitHub.
