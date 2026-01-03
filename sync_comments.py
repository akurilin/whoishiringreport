"""Sync comments from Hacker News 'Who is hiring?' posts.

Standalone script for downloading and incrementally syncing job posting comments.

Usage:
    python sync_comments.py              # Sync last 6 posts (default)
    python sync_comments.py --posts 12   # Sync last 12 posts
    python sync_comments.py --refresh    # Force full refresh (ignore cache)
    python sync_comments.py --output X   # Custom output path

Test-friendly flags:
    python sync_comments.py --post-id 45800465  # Fetch specific post
    python sync_comments.py --max-comments 10   # Limit comments fetched
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
TITLE_PATTERN = re.compile(r"^ask hn: who is hiring\?\s*\(.*\)", re.IGNORECASE)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"
DEFAULT_OUTPUT_PATH = OUT_DIR / "comments.json"
SCHEMA_VERSION = 1
DEFAULT_LAST_SYNC = dt.datetime(2025, 11, 29, tzinfo=dt.UTC).isoformat()


def parse_iso8601(date_str: str) -> dt.datetime | None:
    """Parse ISO8601-ish strings safely to aware UTC datetimes."""
    if not date_str:
        return None
    try:
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _max_date_from_items(items: list[dict], key: str) -> str | None:
    """Return max ISO date string from items[key] values (if parseable)."""
    max_dt = None
    for item in items:
        dt_obj = parse_iso8601(item.get(key, ""))
        if dt_obj and (max_dt is None or dt_obj > max_dt):
            max_dt = dt_obj
    return max_dt.isoformat() if max_dt else None


def normalize_cache_shape(data, kind: str) -> dict:
    """Ensure cache has {items, metadata}; auto-migrate legacy list shape."""
    if isinstance(data, list):
        items = data
        metadata: dict[str, str | None] = {}
    elif isinstance(data, dict):
        items = data.get("items") if isinstance(data.get("items"), list) else []
        metadata = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )
    else:
        items = []
        metadata = {}

    # Derive last_synced_at if missing
    if metadata.get("last_synced_at") is None:
        derived = _max_date_from_items(items, "date") if kind == "comments" else None
        metadata["last_synced_at"] = derived or DEFAULT_LAST_SYNC

    metadata.setdefault("schema_version", SCHEMA_VERSION)

    return {"items": items, "metadata": metadata}


def load_cache(path: Path) -> dict:
    """Load cache file, migrating legacy list shape in-memory if needed."""
    if not path.exists():
        return {
            "items": [],
            "metadata": {
                "last_synced_at": DEFAULT_LAST_SYNC,
                "schema_version": SCHEMA_VERSION,
            },
        }

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    return normalize_cache_shape(raw, "comments")


def write_cache(cache: dict, output_path: Path) -> None:
    """Persist cache ensuring metadata defaults are present."""
    cache = normalize_cache_shape(cache, "comments")
    cache["metadata"].setdefault("last_synced_at", dt.datetime.now(dt.UTC).isoformat())
    cache["metadata"]["schema_version"] = SCHEMA_VERSION

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fetch_hn_item(item_id: int) -> dict | None:
    """Fetch a single item from Hacker News Firebase API."""
    url = f"{HN_API_BASE}/item/{item_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error fetching item {item_id}: {e}")
        return None


def fetch_who_is_hiring_posts(max_posts: int = 6) -> list[dict]:
    """Fetch the N most recent 'Who is hiring?' posts.

    Args:
        max_posts: Maximum number of posts to return (default 6)

    Returns:
        List of post dictionaries sorted by date (newest first)
    """
    page = 0
    posts: list[dict] = []
    seen_ids = set()

    while len(posts) < max_posts:
        params = {
            "query": "Ask HN: Who is hiring?",
            "tags": "story",
            "page": page,
            "hitsPerPage": 50,
        }
        resp = requests.get(SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])

        for hit in hits:
            if len(posts) >= max_posts:
                break

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
            posts.append(
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

    return sorted(posts, key=lambda t: t["created_at"], reverse=True)


def fetch_post_by_id(post_id: int) -> dict | None:
    """Fetch a specific post by ID from Firebase API."""
    post_item = fetch_hn_item(post_id)
    if not post_item:
        return None

    title = (post_item.get("title") or "").strip()
    created_at = dt.datetime.fromtimestamp(
        post_item.get("time", 0), tz=dt.UTC
    ).isoformat()

    return {
        "id": str(post_item.get("id", "")),
        "title": title,
        "author": post_item.get("by", ""),
        "created_at": created_at,
        "hn_url": f"https://news.ycombinator.com/item?id={post_item.get('id', '')}",
        "points": post_item.get("score"),
        "num_comments": len(post_item.get("kids", []) or []),
    }


def fetch_new_comments_algolia(
    post_id: int,
    post_url: str,
    since: dt.datetime | None,
    max_comments: int | None = None,
) -> list[dict]:
    """Fetch new top-level comments for a post via Algolia since a given timestamp.

    Args:
        post_id: The HN post ID
        post_url: URL to the post (stored with each comment)
        since: Only fetch comments created after this timestamp
        max_comments: Optional limit on comments to fetch

    Returns:
        List of comment dictionaries
    """
    params = {
        "tags": f"comment,story_{post_id}",
        "hitsPerPage": 1000,
        "page": 0,
    }
    if since is not None:
        params["numericFilters"] = f"created_at_i>{int(since.timestamp())}"

    comments: list[dict] = []

    while True:
        if max_comments is not None and len(comments) >= max_comments:
            break

        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(
                f"Error fetching comments via Algolia for post {post_id} page {params.get('page')}: {exc}"
            )
            break

        hits = data.get("hits", [])
        for hit in hits:
            if max_comments is not None and len(comments) >= max_comments:
                break

            # Only keep top-level comments (parent is the story)
            if hit.get("parent_id") != post_id:
                continue

            cid = hit.get("objectID")
            cid_str = str(cid) if cid is not None else None
            created_at = hit.get("created_at")
            comments.append(
                {
                    "id": cid_str,
                    "post_url": post_url,
                    "commenter": hit.get("author", ""),
                    "date": created_at,
                    "content": hit.get("comment_text", "") or "",
                }
            )

        page = params.get("page", 0)
        nb_pages = data.get("nbPages", 0)
        if page >= nb_pages - 1:
            break
        params["page"] = page + 1

    return comments


def extract_post_id_from_url(url: str) -> int | None:
    """Extract post ID from Hacker News URL."""
    match = re.search(r"id=(\d+)", url)
    if match:
        return int(match.group(1))
    return None


def sync_comments(
    output_path: Path,
    max_posts: int = 6,
    post_id: int | None = None,
    max_comments: int | None = None,
    refresh: bool = False,
) -> dict:
    """Sync comments from 'Who is hiring?' posts.

    Args:
        output_path: Path to write comments cache
        max_posts: Number of recent posts to sync (ignored if post_id set)
        post_id: Specific post ID to sync (for testing)
        max_comments: Limit total comments fetched (for testing)
        refresh: If True, ignore existing cache

    Returns:
        The merged cache dictionary
    """
    # Load existing cache
    existing_cache = (
        {"items": [], "metadata": {"last_synced_at": None}}
        if refresh
        else load_cache(output_path)
    )
    existing_comments = existing_cache.get("items", [])
    existing_ids = {
        str(c.get("id")) for c in existing_comments if c.get("id") is not None
    }

    last_synced_at = parse_iso8601(
        existing_cache.get("metadata", {}).get("last_synced_at")
    )
    if last_synced_at is None:
        inferred = _max_date_from_items(existing_comments, "date")
        last_synced_at = parse_iso8601(inferred) if inferred else None

    print(
        f"Incremental sync: {len(existing_comments)} existing comments, "
        f"last_synced_at={last_synced_at.isoformat() if last_synced_at else 'none'}"
    )

    # Fetch posts
    if post_id:
        print(f"Fetching specific post: {post_id}")
        post = fetch_post_by_id(post_id)
        if not post:
            print(f"Error: Could not fetch post {post_id}")
            return existing_cache
        posts = [post]
    else:
        print(f"Fetching {max_posts} most recent 'Who is hiring?' posts...")
        posts = fetch_who_is_hiring_posts(max_posts)

    print(f"Found {len(posts)} posts to sync")

    # Fetch comments for each post
    new_comments: list[dict] = []
    comments_fetched = 0

    for i, post in enumerate(posts, 1):
        if max_comments is not None and comments_fetched >= max_comments:
            print("Reached max comment limit; stopping early.")
            break

        post_url = post.get("hn_url", "")
        pid = extract_post_id_from_url(post_url) or int(post.get("id", 0))

        if not pid:
            print(f"Skipping post {i}: Could not extract ID from URL: {post_url}")
            continue

        print(
            f"Fetching comments for post {i}/{len(posts)}: {post.get('title', 'Unknown')}"
        )

        remaining = None
        if max_comments is not None:
            remaining = max_comments - comments_fetched

        post_comments = fetch_new_comments_algolia(
            pid, post_url, last_synced_at, max_comments=remaining
        )

        # Deduplicate against existing
        for c in post_comments:
            cid = str(c.get("id")) if c.get("id") is not None else None
            if cid and cid in existing_ids:
                continue

            new_comments.append(c)
            existing_ids.add(cid or "")
            comments_fetched += 1

            if max_comments is not None and comments_fetched >= max_comments:
                break

        print(f"  Found {len(post_comments)} comments (total new: {len(new_comments)})")

    # Merge and sort
    merged_comments = existing_comments + new_comments
    merged_comments.sort(
        key=lambda c: parse_iso8601(c.get("date", ""))
        or dt.datetime.min.replace(tzinfo=dt.UTC)
    )

    merged_cache = {
        "items": merged_comments,
        "metadata": {
            "last_synced_at": dt.datetime.now(dt.UTC).isoformat(),
            "schema_version": SCHEMA_VERSION,
        },
    }

    # Write cache
    write_cache(merged_cache, output_path)
    print(
        f"\nSaved {len(merged_comments)} comments to {output_path} "
        f"(added {len(new_comments)}, existing {len(existing_comments)})"
    )

    return merged_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync comments from Hacker News 'Who is hiring?' posts."
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=6,
        help="Number of recent posts to sync (default: 6)",
    )
    parser.add_argument(
        "--post-id",
        type=int,
        help="Fetch a specific post ID instead of recent posts (for testing)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        help="Limit total comments fetched (for testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force full refresh, ignoring existing cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sync_comments(
        output_path=args.output,
        max_posts=args.posts,
        post_id=args.post_id,
        max_comments=args.max_comments,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
