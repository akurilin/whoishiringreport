# HN "Who Is Hiring?" tracker

Script to pull recent Hacker News "Who is hiring?" threads, scrape comments, find engineering-management roles, and produce an interactive HTML report.

## Setup
- Requires Python 3.10+.
- Create a venv and install deps (keeps `pip` installs local):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- LLM extraction needs an OpenAI key in `.env`:
  ```
  OPENAI_API_KEY=sk-...
  ```

## Quick start
- One-liner pipeline (fetch posts → comments → matches → extraction → report):
  ```bash
  make all MONTHS=24 \
    POSTS=posts.csv \
    COMMENTS=out/comments.json \
    MATCHES=out/matches.json \
    EXTRACTED=out/matches_with_extraction.json \
    REPORT=out/report.html
  ```
- Open `out/report.html` in your browser to browse matches.
- Adjust `MONTHS` to limit how far back to search. Outputs default to `out/` unless overridden.

## Run steps manually
- Fetch post list (~N months back):  
  `python who_is_hiring.py --months 24 --output posts.csv`
- Fetch comments for those posts:  
  `python who_is_hiring.py --fetch-comments --input posts.csv --output out/comments.json`
- Find engineering-management roles (uses `profiles/engineering_management.yaml`):  
  `python who_is_hiring.py --search-eng-management --input out/comments.json --output out/matches.json`
- Extract structured fields with the LLM (title, location, remote, comp, etc.):  
  `python who_is_hiring.py --extract-from-matches --input out/matches.json --output out/matches_with_extraction.json`
  - Add `--no-extract` on the search step to skip LLM usage entirely.
- Generate the HTML report:  
  `python who_is_hiring.py --generate-html --input out/matches_with_extraction.json --output out/report.html`
- The report opens automatically in your default browser; add `--no-open-report` to skip (e.g., in tests/CI).
- See all flags: `python who_is_hiring.py -h`

## Notes
- If `OPENAI_API_KEY` is missing, extraction is skipped; matches still write but lack enriched fields.
- Cached inputs (`posts.csv`, `out/comments.json`, etc.) let you rerun later steps without re-scraping.
- The script hits the HN Algolia API for post discovery and the official HN API for comments.

## Tests
- Basic end-to-end check:  
  ```bash
  make test-e2e
  ```
