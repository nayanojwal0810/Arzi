# Arzi — Financial Complaint Intelligence

Arzi analyzes financial complaints to help banking operations teams understand recurring customer problems. It brings together complaint details and written customer descriptions so analysts can study patterns and find issues that need attention.

## Data

The project uses 117,214 historical complaints from the US Consumer Financial Protection Bureau (CFPB), dated January 2019 to January 2020. Each record includes information such as the financial product, complaint issue, company, response, and customer narrative.

Download the [historical complaint dataset](https://raw.githubusercontent.com/EmilHvitfeldt/smltar/3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz) and save it as `data/raw/complaints.csv.gz`. The raw dataset is kept locally and is not uploaded to GitHub.

## Getting Started

Use Python 3.11 or newer. From the project root, run:

```bash
python -m pip install -e ".[dev]"
python -c "import arzi; print(arzi.__version__)"
```

## Project Structure

- `src/arzi/` — Python application code.
- `data/raw/` — locally stored complaint dataset, excluded from Git.
