# Instructions

- Make sure to have the .venv sourced if working with python.
- Update requirements.txt once python dependencies are updated

## File map (quick reference)
- who_is_hiring.py — main script to fetch HN "Who is hiring" threads, pull comments, search for matches, and generate the HTML report.
- templates/report.html — Jinja2 template for the interactive HTML report.
- requirements.txt — Python dependencies lock for the script.
- posts.csv — cached list of HN "Who is hiring" posts (id, title, URL, etc.).
- comments.json — cached comments scraped from posts.
- matches.json — filtered engineering-management matches from comments.
- matches_with_extraction.json — matches enriched with extracted fields for the report.
- extraction_proposal.md — notes on extraction approach/plan.
- engineering_management_search_plan.md — notes on search strategy/plan.
