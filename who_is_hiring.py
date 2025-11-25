"""Fetch recent 'Ask HN: Who is hiring?' threads and save them to CSV.

Run with: python who_is_hiring.py --months 24 --output who_is_hiring_posts.csv
Or fetch comments: python who_is_hiring.py --fetch-comments --input posts.csv --output comments.json
"""

import argparse
import csv
import datetime as dt
import json
import re
from typing import Dict, Iterable, List, Optional

import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
# Match the canonical monthly thread titles like "Ask HN: Who is hiring? (November 2025)"
TITLE_PATTERN = re.compile(r"^ask hn: who is hiring\?\s*\(.*\)", re.IGNORECASE)
# Extract post ID from HN URL
ID_FROM_URL_PATTERN = re.compile(r"id=(\d+)")


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


def fetch_hn_item(item_id: int) -> Optional[Dict]:
    """Fetch a single item from Hacker News API."""
    url = f"{HN_API_BASE}/item/{item_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error fetching item {item_id}: {e}")
        return None


def fetch_comment_and_replies(
    comment_id: int, post_url: str, comments: List[Dict]
) -> None:
    """Recursively fetch a comment and all its replies."""
    comment_item = fetch_hn_item(comment_id)
    if not comment_item:
        return

    # Skip deleted/dead comments
    if comment_item.get("deleted") or comment_item.get("dead"):
        return

    # Extract comment data
    commenter = comment_item.get("by", "")
    time_ts = comment_item.get("time", 0)
    # Convert Unix timestamp to ISO format
    comment_date = dt.datetime.fromtimestamp(time_ts, tz=dt.timezone.utc).isoformat()
    text = comment_item.get("text", "")

    # Add comment to list
    comments.append(
        {
            "post_url": post_url,
            "commenter": commenter,
            "date": comment_date,
            "content": text,
        }
    )

    # Recursively fetch replies
    kids = comment_item.get("kids", [])
    for kid_id in kids:
        fetch_comment_and_replies(kid_id, post_url, comments)


def fetch_all_comments(post_id: int, post_url: str, comments: List[Dict]) -> None:
    """Fetch all comments from a post and append to comments list."""
    post_item = fetch_hn_item(post_id)
    if not post_item:
        return

    # Get top-level comment IDs
    kids = post_item.get("kids", [])
    if not kids:
        return

    # Fetch all comments recursively
    for kid_id in kids:
        fetch_comment_and_replies(kid_id, post_url, comments)


def extract_post_id_from_url(url: str) -> Optional[int]:
    """Extract post ID from Hacker News URL."""
    match = ID_FROM_URL_PATTERN.search(url)
    if match:
        return int(match.group(1))
    return None


def read_posts_csv(csv_path: str) -> List[Dict]:
    """Read posts from CSV and return list of post data."""
    posts = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            posts.append(row)
    return posts


def fetch_comments_from_posts(csv_path: str) -> List[Dict]:
    """Fetch all comments from posts listed in CSV file."""
    posts = read_posts_csv(csv_path)
    all_comments = []

    for i, post in enumerate(posts, 1):
        post_url = post.get("hn_url", "")
        post_id = extract_post_id_from_url(post_url)

        if not post_id:
            print(f"Skipping post {i}: Could not extract ID from URL: {post_url}")
            continue

        print(
            f"Fetching comments for post {i}/{len(posts)}: {post.get('title', 'Unknown')} (ID: {post_id})"
        )
        post_comments = []
        fetch_all_comments(post_id, post_url, post_comments)
        all_comments.extend(post_comments)
        print(
            f"  Found {len(post_comments)} comments (total so far: {len(all_comments)})"
        )

    return all_comments


def write_json(data: List[Dict], output_path: str) -> None:
    """Write data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 'Ask HN: Who is hiring?' threads and save them to CSV, or fetch comments from posts."
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Mode: fetch comments from posts in CSV file instead of fetching post URLs.",
    )
    parser.add_argument(
        "--input",
        default="posts.csv",
        help="Input CSV file (for --fetch-comments mode, default: posts.csv).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=24,
        help="How many months back to search (approximate, default: 24). Only used in default mode.",
    )
    parser.add_argument(
        "--output",
        default="who_is_hiring_posts.csv",
        help="Path to write output (default: who_is_hiring_posts.csv for posts, comments.json for comments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fetch_comments:
        # Mode 2: Fetch comments from posts in CSV
        if not args.output.endswith(".json"):
            # Default to .json if not specified
            if args.output == "who_is_hiring_posts.csv":
                args.output = "comments.json"

        comments = fetch_comments_from_posts(args.input)
        write_json(comments, args.output)
        print(f"\nWrote {len(comments)} comments to {args.output}")
    else:
        # Mode 1: Fetch post URLs (original functionality)
        since_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=31 * args.months
        )
        threads = fetch_who_is_hiring_threads(since_date)
        write_csv(threads, args.output)
        print(f"Wrote {len(threads)} threads to {args.output}")


if __name__ == "__main__":
    main()
