.PHONY: sync report all test baml-generate \
        extract-instructor-openai extract-instructor-gemini \
        extract-baml-openai extract-baml-gemini \
        eval-instructor-openai eval-instructor-gemini eval-instructor-all \
        eval-baml-openai eval-baml-gemini \
        eval-all-permutations

# --- VARIABLES ---

PYTHON := .venv/bin/python
ACTIVATE := . .venv/bin/activate &&
EVAL_FLAGS := -v --tb=no -W ignore::DeprecationWarning -W ignore::FutureWarning --no-header

ALL_MODELS := gpt-4o-mini,gemini-2.0-flash-lite,gemini-2.5-flash-lite
ALL_EXTRACTORS := instructor,baml

# --- DATA PIPELINE ---

sync:
	$(ACTIVATE) $(PYTHON) sync_comments.py

report:
	$(ACTIVATE) $(PYTHON) generate_report.py

all: sync extract-instructor-openai report

# --- EXTRACTION ---

extract-instructor-openai:
	$(ACTIVATE) $(PYTHON) extract_jobs.py

extract-instructor-gemini:
	$(ACTIVATE) $(PYTHON) extract_jobs.py --model gemini-2.0-flash-lite

extract-baml-openai:
	$(ACTIVATE) $(PYTHON) extract_jobs.py --extractor baml

extract-baml-gemini:
	$(ACTIVATE) $(PYTHON) extract_jobs.py --extractor baml --model gemini-2.0-flash-lite

# --- EVAL SUITE ---

eval-instructor-openai:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS)

eval-instructor-gemini:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS) --models gemini-2.0-flash-lite

eval-instructor-all:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS) --models $(ALL_MODELS)

eval-baml-openai:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS) --extractors baml

eval-baml-gemini:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS) --extractors baml --models gemini-2.0-flash-lite

eval-all-permutations:
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_extraction.py $(EVAL_FLAGS) --extractors $(ALL_EXTRACTORS) --models $(ALL_MODELS)

# --- TESTING & BUILD ---

test:
	$(ACTIVATE) $(PYTHON) -m pytest tests/ -v

baml-generate:
	$(ACTIVATE) baml-cli generate
