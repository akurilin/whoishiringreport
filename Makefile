.PHONY: posts comments matches extract report all test-e2e

PYTHON := .venv/bin/python
ACTIVATE := . .venv/bin/activate &&
MONTHS ?= 24
POSTS ?= posts.csv
COMMENTS ?= comments.json
MATCHES ?= matches.json
EXTRACTED ?= matches_with_extraction.json
REPORT ?= out/report.html
TEST_OUT ?= out/test
TEST_POST_ID ?=

posts:
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --months $(MONTHS) --output $(POSTS)

comments:
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --fetch-comments --input $(POSTS) --output $(COMMENTS)

matches:
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --search-eng-management --input $(COMMENTS) --output $(MATCHES)

extract:
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --extract-from-matches --input $(MATCHES) --output $(EXTRACTED)

report:
	@mkdir -p $(dir $(REPORT))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --generate-html --input $(EXTRACTED) --output $(REPORT)

all: posts comments matches extract report

test-e2e:
	@mkdir -p $(TEST_OUT)
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_e2e.py -q
