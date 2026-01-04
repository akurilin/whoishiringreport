# HN "Who Is Hiring?" Job Extractor

Extract structured job data from Hacker News "Who is hiring?" threads and generate an interactive HTML report.

## Report preview
![Interactive report screenshot](docs/assets/report.png)

## Quick start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here  # or add to .env file

# Run the pipeline
python sync_comments.py           # Fetch comments from recent threads
python extract_jobs.py --limit 50 # Extract job data (limit for testing)
python generate_report.py         # Generate HTML report (opens in browser)
```

## Scripts

### 1. `sync_comments.py` - Fetch comments from HN

```bash
python sync_comments.py                    # Fetch from recent threads (default: 6 months)
python sync_comments.py --max-posts 3      # Limit to 3 most recent threads
python sync_comments.py --post-id 12345    # Fetch specific thread by ID
python sync_comments.py --refresh          # Re-fetch all comments
```

Output: `out/comments.json`

### 2. `extract_jobs.py` - Extract structured job data

Uses GPT-4o-mini to extract structured fields from each comment:
- Role title, company name, company stage
- Location, remote status, employment type
- Salary range, equity percentage
- Application method, company URL

```bash
python extract_jobs.py                     # Extract all unprocessed comments
python extract_jobs.py --limit 100         # Extract first 100 unprocessed
python extract_jobs.py --refresh           # Re-extract all comments
python extract_jobs.py --model gpt-4o      # Use different model
```

Output: `out/extracted_jobs.json`

### 3. `generate_report.py` - Generate HTML report

```bash
python generate_report.py                  # Generate and open in browser
python generate_report.py --no-browser     # Generate without opening
```

Output: `out/report.html`

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` environment variable (or in `.env` file)

## Tests

```bash
pytest tests/ -v
```

The extraction tests require `OPENAI_API_KEY` and make real API calls.
