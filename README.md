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
python extract_jobs.py --extractor baml    # Use BAML extractor
```

Output: `out/extracted_jobs.json`

#### Extractors

The project supports two extraction backends:

| Extractor | Description | Usage |
|-----------|-------------|-------|
| **Instructor** (default) | Uses [Instructor](https://github.com/jxnl/instructor) with Pydantic models for structured output. Mature, well-tested. | `--extractor instructor` |
| **BAML** | Uses [BAML](https://docs.boundaryml.com/) domain-specific language for LLM extraction. Declarative schema with type-safe outputs. | `--extractor baml` |

Both extractors produce identical output format and can be compared using the eval suite.

#### Model Comparison

| Extractor | Model | Avg Time | Accuracy | Tokens |
|-----------|-------|----------|----------|--------|
| instructor | gpt-4o-mini | 3.06s | 11/11 | 15,603 |
| instructor | gemini-2.0-flash-lite | 1.65s | 11/11 | 13,714 |
| instructor | gemini-2.5-flash-lite | 1.04s | 10/11 | 17,715 |
| baml | gpt-4o-mini | 5.83s | 11/11 | 15,176 |
| baml | gemini-2.0-flash-lite | 2.15s | 11/11 | 17,733 |
| baml | gemini-2.5-flash-lite | 1.37s | 11/11 | 17,863 |

Provider is auto-detected from model name. Set `GEMINI_API_KEY` in `.env` for Gemini models.

#### Known Model Limitations

**instructor::gemini-2.5-flash-lite** fails on:
- `single_role_complete_data`: Returns `equity: "unknown"` instead of `null` when equity is mentioned without a specific percentage

**Root cause:** Gemini models occasionally return placeholder values (`"unknown"`, `0`) rather than `null` when data is mentioned but not fully specified. The BAML extractor handles this more reliably.

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
pytest tests/test_extraction.py -v --models gemini-2.0-flash-lite  # Eval suite (Gemini)
pytest tests/test_extraction.py -v --extractors baml    # Eval suite (BAML)
pytest tests/test_extraction.py -v --extractors instructor,baml --models gpt-4o-mini,gemini-2.0-flash-lite  # Compare all
```

The extraction tests require API keys and make real API calls.

### Eval Statistics

Each eval run automatically persists detailed statistics to `out/eval_stats.jsonl`. This enables tracking extraction quality and performance over time.

**Analyze stats:**
```bash
python analyze_stats.py                    # Analyze eval_stats.jsonl
```

**Statistics tracked per extraction:**
- Model and extractor used
- Elapsed time and token usage
- Success/failure status
- Number of roles extracted
- Error type (if failed)

**Analysis output includes:**
- Success rates by model/extractor
- Average extraction time comparison
- Token efficiency (tokens per role)
- Slowest extractions
- Error breakdown by type
- Run history across sessions

This data helps identify:
- Which model+extractor combinations perform best
- Performance regressions after changes
- Which test cases are consistently problematic
- Cost/speed tradeoffs between configurations
