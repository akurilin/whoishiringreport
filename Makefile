.PHONY: sync extract extract-gemini report all test eval eval-gemini

PYTHON := .venv/bin/python
ACTIVATE := . .venv/bin/activate &&

# Sync comments from HN threads
sync:
	$(ACTIVATE) $(PYTHON) sync_comments.py

# Extract structured job data (default: OpenAI)
extract:
	$(ACTIVATE) $(PYTHON) extract_jobs.py

# Extract with Gemini
extract-gemini:
	$(ACTIVATE) $(PYTHON) extract_jobs.py --model gemini-2.0-flash-lite

# Generate HTML report
report:
	$(ACTIVATE) $(PYTHON) generate_report.py

# Run full pipeline
all: sync extract report

# Run tests
test:
	$(ACTIVATE) $(PYTHON) -m pytest tests/ -v

# Run eval suite with OpenAI (default)
eval:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py -v

# Run eval suite with Gemini
eval-gemini:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py -v --models gemini-2.0-flash-lite

# Compare all models
eval-compare:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py -v --models gpt-4o-mini,gemini-2.0-flash-lite,gemini-2.5-flash-lite
