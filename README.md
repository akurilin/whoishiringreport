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

Uses LLM to extract structured fields from each comment:
- Role title, company name, company stage
- Location, remote status, employment type
- Salary range, equity percentage
- Application method, company URL

```bash
python extract_jobs.py                     # Extract all unprocessed comments
python extract_jobs.py --limit 100         # Extract first 100 unprocessed
python extract_jobs.py --refresh           # Re-extract all comments
python extract_jobs.py --model gpt-4o      # Use different OpenAI model
python extract_jobs.py --model gemini-2.0-flash-lite  # Use Gemini (faster)
```

Output: `out/extracted_jobs.json`

#### Model Comparison

| Model | Avg Time/Extraction | Accuracy | Notes |
|-------|---------------------|----------|-------|
| gpt-4o-mini | 3.51s | 11/11 tests | Default |
| gemini-2.0-flash-lite | 1.68s | 10/11 tests | ~2x faster |
| gemini-2.5-flash-lite | 1.17s | 9/11 tests | ~3x faster, least accurate |
| gemini-3-flash-preview | 5.04s | 11/11 tests | Most accurate Gemini, slower |

Provider is auto-detected from model name. Set `GEMINI_API_KEY` in `.env` for Gemini models.

#### Known Model Limitations

**gemini-2.0-flash-lite** fails on:
- `single_role_complete_data`: Returns `equity: "unknown"` instead of `null` when equity is mentioned without a specific percentage
- `contract_remote`: Returns `salary_max: 0` instead of `null` when salary is not specified

**gemini-2.5-flash-lite** fails on:
- `single_role_complete_data`: Same `equity: "unknown"` issue as above
- `specific_equity_high_salary`: Fails to extract the second role (Applied AI Engineer) from a multi-role posting

**Root cause:** Gemini models tend to return placeholder values (`"unknown"`, `0`) rather than `null` when data is mentioned but not fully specified. The schema explicitly instructs to use `null` in these cases, but Gemini models don't follow this instruction consistently.

### 3. `generate_report.py` - Generate HTML report

```bash
python generate_report.py                  # Generate and open in browser
python generate_report.py --no-browser     # Generate without opening
```

Output: `out/report.html`

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` environment variable (or in `.env` file)
- `GEMINI_API_KEY` (optional, for Gemini models)

## Tests

```bash
pytest tests/ -v                                        # Run all tests
pytest tests/test_extraction.py -v                      # Eval suite (OpenAI)
pytest tests/test_extraction.py -v --model gemini-2.0-flash-lite  # Eval suite (Gemini)
```

The extraction tests require API keys and make real API calls.
