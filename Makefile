.PHONY: sync extract report all test

PYTHON := .venv/bin/python
ACTIVATE := . .venv/bin/activate &&

# Sync comments from HN threads
sync:
	$(ACTIVATE) $(PYTHON) sync_comments.py

# Extract structured job data
extract:
	$(ACTIVATE) $(PYTHON) extract_jobs.py

# Generate HTML report
report:
	$(ACTIVATE) $(PYTHON) generate_report.py

# Run full pipeline
all: sync extract report

# Run tests
test:
	$(ACTIVATE) $(PYTHON) -m pytest tests/ -v
