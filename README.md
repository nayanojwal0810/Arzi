# Arzi — Financial Complaint Intelligence

Arzi is a local analytics project for exploring historical US consumer financial complaints using PostgreSQL, Python, statistics, and text analysis. It is designed to surface patterns for investigation, not score financial institutions.

## Dataset

The project uses a fixed set of **117,214 historical Consumer Financial Protection Bureau (CFPB) complaints**, dated **January 1, 2019–January 9, 2020**. January 2020 is incomplete and is excluded from full-period statistical comparisons.

Download the [historical complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) into `data/raw/complaints.csv.gz` without modifying the compressed file. Raw complaint text and generated model artifacts are not committed to GitHub.

## Run Locally

Requires Python 3.11+, a running PostgreSQL instance, and locally configured PostgreSQL authentication.

```bash
python -m pip install -e ".[dev]"
python scripts/load_data.py
python scripts/run_analysis.py
python scripts/run_statistics.py
python scripts/run_nlp.py
python -m pytest -q
```

Run `python scripts/run_nlp.py --retrain` only when intentionally rebuilding the topic model. The saved model is checked against the source-data fingerprint and its configuration before reuse.

## What the Project Analyzes

- **SQL and EDA:** Complaint volumes by product, issue, company, response, submission channel, and month.
- **Statistics:** Recorded timely-response differences between products, plus changes in the product mix between the two halves of 2019, using confidence intervals and multiple-testing correction.
- **NLP:** TF-IDF and non-negative matrix factorization (NMF) discover ten provisional topics from consumer narratives. Each complaint ID is retained, including those without a usable topic assignment.

## Selected Findings

- Credit reporting accounts for **55,558 complaints (47.40%)** in this historical collection.
- **115,136 of 117,214 complaints (98.23%)** have a recorded timely response. Debt collection and payday/title/personal loans show lower rates than the other products combined.
- Credit reporting's share increased from **45.34% in the first half of 2019** to **49.33% in the second half**.
- The NLP pipeline assigned topics to **117,107 complaints** and left **107 unassigned**. One small topic predominantly captures repeated dispute-letter wording; another mostly captures institution-specific language. These are limitations of the model, not verified business issue categories.

Complaint counts are not institution complaint rates: the dataset has no customer or transaction denominators. The observed relationships do not prove misconduct or causation. Recorded timely response is not a measure of customer satisfaction. NLP themes are exploratory and should be interpreted alongside the original product and issue labels.

## Project Structure

- `src/arzi/` — ingestion, PostgreSQL access, analysis, statistics, and NLP.
- `sql/` — database schema, verification queries, and analytical views.
- `scripts/` — executable local workflows.
- `tests/` — synthetic Python tests and validation checks.

Local plots, aggregate JSON summaries, and trained models are generated under `data/processed/` and excluded from version control.
