.PHONY: posts comments matches extract report all test-e2e list-profiles

PYTHON := .venv/bin/python
ACTIVATE := . .venv/bin/activate &&
MONTHS ?= 6
POSTS ?= out/posts.json
COMMENTS ?= out/comments.json
PROFILE ?=
# Derive a slug from the profile filename (or default to engineering_management)
PROFILE_SLUG := $(if $(PROFILE),$(basename $(notdir $(PROFILE))),engineering_management)
MATCHES ?= out/$(PROFILE_SLUG)/matches.json
EXTRACTED ?= out/$(PROFILE_SLUG)/matches_with_extraction.json
REPORT ?= out/$(PROFILE_SLUG)/report.html
TEST_OUT ?= out/test
TEST_POST_ID ?=
REFRESH_CACHE ?=

posts:
	@mkdir -p $(dir $(POSTS))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --months $(MONTHS) --output $(POSTS) $(if $(REFRESH_CACHE),--refresh-cache,)

comments:
	@mkdir -p $(dir $(COMMENTS))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --fetch-comments --input $(POSTS) --output $(COMMENTS) $(if $(REFRESH_CACHE),--refresh-cache,)

matches:
	@mkdir -p $(dir $(MATCHES))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --search-eng-management --input $(COMMENTS) --output $(MATCHES) $(if $(PROFILE),--profile $(PROFILE),)

extract:
	@mkdir -p $(dir $(EXTRACTED))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --extract-from-matches --input $(MATCHES) --output $(EXTRACTED)

report:
	@mkdir -p $(dir $(REPORT))
	$(ACTIVATE) $(PYTHON) who_is_hiring.py --generate-html --input $(EXTRACTED) --output $(REPORT)

all: posts comments matches extract report

test-e2e:
	@mkdir -p $(TEST_OUT)
	$(ACTIVATE) $(PYTHON) -m pytest tests/test_e2e.py -q

list-profiles:
	@echo "Available profiles:"
	@ls profiles/*.yaml 2>/dev/null || echo "  (none found)"
