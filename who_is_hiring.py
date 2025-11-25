"""Fetch recent 'Ask HN: Who is hiring?' threads and save them to CSV.

Run with: python who_is_hiring.py --months 24 --output who_is_hiring_posts.csv
"""
import argparse
import csv
import datetime as dt
import re
from typing import Dict, Iterable, List

import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
# Match the canonical monthly thread titles like "Ask HN: Who is hiring? (November 2025)"
TITLE_PATTERN = re.compile(r"^ask hn: who is hiring\?\s*\(.*\)", re.IGNORECASE)


def fetch_who_is_hiring_threads(since: dt.datetime) -> List[Dict]:
    """Return threads matching the title pattern since the given UTC datetime."""
    since_ts = int(since.timestamp())
    page = 0
    threads: List[Dict] = []
    seen_ids = set()

    while True:
        params = {
            "query": "Ask HN: Who is hiring?",
            "tags": "story",
            "numericFilters": f"created_at_i>{since_ts}",
            "page": page,
            "hitsPerPage": 50,
        }
        resp = requests.get(SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hits: Iterable[Dict] = data.get("hits", [])

        for hit in hits:
            title = (hit.get("title") or "").strip()
            if (hit.get("author") or "").lower() != "whoishiring":
                continue
            if not TITLE_PATTERN.match(title):
                continue

            hn_id = hit.get("objectID")
            if hn_id in seen_ids:
                continue
            seen_ids.add(hn_id)

            created_at = hit.get("created_at")
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hn_id}"
            threads.append(
                {
                    "id": hn_id,
                    "title": title,
                    "author": hit.get("author"),
                    "created_at": created_at,
                    "hn_url": url,
                    "points": hit.get("points"),
                    "num_comments": hit.get("num_comments"),
                }
            )

        if page >= (data.get("nbPages", 0) - 1):
            break
        page += 1

    return sorted(threads, key=lambda t: t["created_at"], reverse=True)


def write_csv(rows: Iterable[Dict], output_path: str) -> None:
    fieldnames = [
        "id",
        "title",
        "author",
        "created_at",
        "hn_url",
        "points",
        "num_comments",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 'Ask HN: Who is hiring?' threads and save them to CSV."
    )
    parser.add_argument(
        "--months",
        type=int,
        default=24,
        help="How many months back to search (approximate, default: 24).",
    )
    parser.add_argument(
        "--output",
        default="who_is_hiring_posts.csv",
        help="Path to write CSV output (default: who_is_hiring_posts.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Use a simple month approximation (31 days) to stay on the safe side.
    since_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=31 * args.months)
    threads = fetch_who_is_hiring_threads(since_date)
    write_csv(threads, args.output)
    print(f"Wrote {len(threads)} threads to {args.output}")


if __name__ == "__main__":
    main()
