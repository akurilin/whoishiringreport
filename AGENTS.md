## File map (quick reference)

- `sync_comments.py` - Fetches comments from HN "Who is hiring?" threads
- `extract_jobs.py` - LLM extraction of structured job data (OpenAI/Gemini)
- `generate_report.py` - Generates HTML report from extracted jobs
- `tests/conftest.py` - Pytest config, timing infrastructure, eval hooks
- `tests/test_extraction.py` - Eval suite for extraction accuracy
- `tests/fixtures/eval_cases.json` - Golden test cases for eval suite
- `Makefile` - Commands: sync, extract, report, eval, eval-compare
