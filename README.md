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
- Make sure your virtualenv is created and activated, and dependencies installed (`pip install -r requirements.txt`), before running `make all`.
- One-liner pipeline (fetch posts → comments → matches → extraction → report):
  ```bash
  make all MONTHS=6 \
    POSTS=out/posts.json \
    COMMENTS=out/comments.json \
    PROFILE=profiles/engineering_management.yaml \
    MATCHES=out/engineering_management/matches.json \
    EXTRACTED=out/engineering_management/matches_with_extraction.json \
    REPORT=out/engineering_management/report.html
  ```
- Swap profiles without clobbering others:
  ```bash
  make all PROFILE=profiles/ux_designer.yaml
  ```
- Open `out/report.html` in your browser to browse matches.
- Adjust `MONTHS` to limit how far back to search. Outputs default to `out/` unless overridden.

## Run steps manually
- Fetch post list (~N months back):  
  `python who_is_hiring.py --months 6 --output out/posts.json`
- Fetch comments for those posts:  
  `python who_is_hiring.py --fetch-comments --input out/posts.json --output out/comments.json`
- Find engineering-management roles (uses `profiles/engineering_management.yaml` by default):  
  `python who_is_hiring.py --search-eng-management --input out/comments.json --output out/engineering_management/matches.json`
- Find UX/design roles (swap in the UX profile):  
  `python who_is_hiring.py --search-eng-management --profile profiles/ux_designer.yaml --input out/comments.json --output out/ux_designer/matches.json`
- Extract structured fields with the LLM (title, location, remote, comp, etc.):  
  `python who_is_hiring.py --extract-from-matches --input out/engineering_management/matches.json --output out/engineering_management/matches_with_extraction.json`
  - Add `--no-extract` on the search step to skip LLM usage entirely.
- Generate the HTML report:  
  `python who_is_hiring.py --generate-html --input out/engineering_management/matches_with_extraction.json --output out/engineering_management/report.html`
- The report opens automatically in your default browser; add `--no-open-report` to skip (e.g., in tests/CI).
- Pass `--refresh-cache` to the post or comment steps to force re-downloads; otherwise cached `posts.json`/`comments.json` will be reused.
- Provide `PROFILE=profiles/ux_designer.yaml` (or any other profile) to `make all` to run the pipeline with an alternate role profile; defaults route per-profile outputs into `out/{profile_stem}/`.
- List available profiles: `make list-profiles`
- See all flags: `python who_is_hiring.py -h`

## Profile-specific runs
- End-to-end via Make (profile-specific outputs land in `out/<profile_stem>/`):  
  `make all PROFILE=profiles/ux_designer.yaml`
- End-to-end manually (swap profile/output paths as needed):  
  1) `python who_is_hiring.py --months 6 --output out/posts.json`  
  2) `python who_is_hiring.py --fetch-comments --input out/posts.json --output out/comments.json`  
  3) `python who_is_hiring.py --search-eng-management --profile profiles/ux_designer.yaml --input out/comments.json --output out/ux_designer/matches.json`  
  4) `python who_is_hiring.py --extract-from-matches --input out/ux_designer/matches.json --output out/ux_designer/matches_with_extraction.json`  
  5) `python who_is_hiring.py --generate-html --input out/ux_designer/matches_with_extraction.json --output out/ux_designer/report.html --no-open-report`

## Profiles (what they are and how to create one)
- Profiles live in `profiles/*.yaml` and define regex patterns for the roles you want to catch (e.g., engineering management, UX/design).
- To add a profile:
  1) Copy an existing file in `profiles/` and rename it (e.g., `profiles/data_science.yaml`).
  2) Edit the `patterns:` list with your regexes and optional `name` fields.
  3) Run `make list-profiles` to confirm it’s picked up, then run the pipeline with `PROFILE=profiles/your_profile.yaml`.
- Regexes can be tricky; the fastest path is to describe your target roles/patterns to an AI coding assistant/agent and let it draft or refine the profile YAML for you, then review/tweak as needed.

## Notes
- If `OPENAI_API_KEY` is missing, extraction is skipped; matches still write but lack enriched fields.
- Cached inputs (`out/posts.json`, `out/comments.json`, etc.) let you rerun later steps without re-scraping; the script now reuses them by default. Per-profile outputs land in `out/{profile_stem}/` to avoid clobbering other profiles.
- The script hits the HN Algolia API for post discovery and the official HN API for comments.

## Tests
- Basic end-to-end check:  
  ```bash
  make test-e2e
  ```
