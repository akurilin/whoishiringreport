"""Fetch recent 'Ask HN: Who is hiring?' threads and save them to JSON.

Run with: python who_is_hiring.py --months 6 --output posts.json
Or fetch comments: python who_is_hiring.py --fetch-comments --input posts.json --output out/comments.json
Or search roles: python who_is_hiring.py --search --profile profiles/engineering_management.yaml --input out/comments.json --output out/engineering_management/matches.json
Or extract from matches: python who_is_hiring.py --extract-from-matches --input out/engineering_management/matches.json --output out/engineering_management/matches_with_extraction.json
Or generate HTML report: python who_is_hiring.py --generate-html --input out/engineering_management/matches_with_extraction.json --output out/engineering_management/report.html
"""

import argparse
import csv
import datetime as dt
import html
import json
import os
from html.parser import HTMLParser
import re
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
# Match the canonical monthly thread titles like "Ask HN: Who is hiring? (November 2025)"
TITLE_PATTERN = re.compile(r"^ask hn: who is hiring\?\s*\(.*\)", re.IGNORECASE)
# Extract post ID from HN URL
ID_FROM_URL_PATTERN = re.compile(r"id=(\d+)")
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"
DEFAULT_PROFILE_PATH = BASE_DIR / "profiles" / "engineering_management.yaml"
DEFAULT_POSTS_PATH = OUT_DIR / "posts.json"
DEFAULT_POSTS_OUTPUT = DEFAULT_POSTS_PATH
DEFAULT_COMMENTS_PATH = OUT_DIR / "comments.json"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
SCHEMA_VERSION = 1
DEFAULT_LAST_SYNC = dt.datetime(2025, 11, 29, tzinfo=dt.timezone.utc).isoformat()


class AnchorPreservingSanitizer(HTMLParser):
    """Escape HTML while allowing safe <a> tags.

    HN comments sometimes include anchor tags (e.g., markdown links rendered by HN).
    We want those clickable in the report without allowing arbitrary HTML. This parser
    escapes everything except http(s)/mailto anchors, which are re-emitted with
    target/rel safety attributes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    @staticmethod
    def _is_safe_href(href: str) -> bool:
        lowered = href.lower()
        return (
            lowered.startswith("http://")
            or lowered.startswith("https://")
            or lowered.startswith("mailto:")
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "a":
            href = ""
            for name, value in attrs:
                if name.lower() == "href" and value:
                    href = value
                    break
            if href and self._is_safe_href(href):
                safe_href = html.escape(href, quote=True)
                self.parts.append(
                    f'<a href="{safe_href}" target="_blank" rel="nofollow noreferrer">'
                )
                return

        self.parts.append(html.escape(self.get_starttag_text()))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self.parts.append("</a>")
        else:
            self.parts.append(html.escape(f"</{tag}>"))

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        # Escape self-closing tags (e.g., <br/>) to avoid rendering unintended HTML
        self.parts.append(html.escape(self.get_starttag_text()))

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:
        # Preserve original entities instead of decoding to keep intent clear
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")


def sanitize_full_content(content: str) -> str:
    """Return HTML-safe content while keeping anchors clickable."""

    if not content:
        return ""

    parser = AnchorPreservingSanitizer()
    parser.feed(content)
    parser.close()
    sanitized = "".join(parser.parts)
    return sanitized.replace("\n", "<br>")


def fetch_who_is_hiring_threads(
    since: dt.datetime, max_posts: Optional[int] = None
) -> List[Dict]:
    """Return threads matching the title pattern since the given UTC datetime.

    max_posts can be used to cap how many recent threads are returned (for tests).
    """
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

    threads = sorted(threads, key=lambda t: t["created_at"], reverse=True)

    if max_posts is not None:
        threads = threads[:max_posts]

    return threads


def write_posts_json(rows: Iterable[Dict], output_path: str) -> None:
    """Write posts to JSON in a consistent schema."""
    cache = {
        "items": list(rows),
        "metadata": {
            "last_synced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        },
    }
    write_json(cache, output_path)


def profile_slug(profile_path: Optional[str]) -> str:
    """Create a safe slug from a profile path (basename without extension)."""
    if not profile_path:
        return "engineering_management"
    slug = Path(profile_path).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)
    return slug or "profile"


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
    comment_id: int,
    post_url: str,
    comments: List[Dict],
    max_comments_per_post: Optional[int] = None,
    global_counter: Optional[List[int]] = None,
    max_comments_total: Optional[int] = None,
) -> None:
    """Recursively fetch a comment and all its replies."""

    if max_comments_per_post is not None and len(comments) >= max_comments_per_post:
        return
    if (
        max_comments_total is not None
        and global_counter is not None
        and global_counter[0] >= max_comments_total
    ):
        return
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

    if global_counter is not None:
        global_counter[0] += 1

    if max_comments_per_post is not None and len(comments) >= max_comments_per_post:
        return
    if (
        max_comments_total is not None
        and global_counter is not None
        and global_counter[0] >= max_comments_total
    ):
        return

    # Recursively fetch replies
    kids = comment_item.get("kids", [])
    for kid_id in kids:
        fetch_comment_and_replies(
            kid_id,
            post_url,
            comments,
            max_comments_per_post=max_comments_per_post,
            global_counter=global_counter,
            max_comments_total=max_comments_total,
        )


def fetch_all_comments(
    post_id: int,
    post_url: str,
    comments: List[Dict],
    max_comments_per_post: Optional[int] = None,
    global_counter: Optional[List[int]] = None,
    max_comments_total: Optional[int] = None,
) -> None:
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
        if max_comments_per_post is not None and len(comments) >= max_comments_per_post:
            break
        if (
            max_comments_total is not None
            and global_counter is not None
            and global_counter[0] >= max_comments_total
        ):
            break
        fetch_comment_and_replies(
            kid_id,
            post_url,
            comments,
            max_comments_per_post=max_comments_per_post,
            global_counter=global_counter,
            max_comments_total=max_comments_total,
        )


def fetch_new_comments_algolia(
    post_id: int, post_url: str, since: Optional[dt.datetime]
) -> List[Dict]:
    """Fetch new top-level comments for a post via Algolia since a given timestamp."""

    params = {
        "tags": f"comment,story_{post_id}",
        "hitsPerPage": 1000,
        "page": 0,
    }
    if since is not None:
        params["numericFilters"] = f"created_at_i>{int(since.timestamp())}"

    comments: List[Dict] = []

    while True:
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:  # noqa: BLE001
            print(
                f"Error fetching comments via Algolia for post {post_id} page {params.get('page')}: {exc}"
            )
            break

        hits = data.get("hits", [])
        for hit in hits:
            # Only keep top-level comments (parent is the story)
            if hit.get("parent_id") != post_id:
                continue

            cid = hit.get("objectID")
            created_at = hit.get("created_at")
            comments.append(
                {
                    "id": int(cid) if cid is not None and str(cid).isdigit() else cid,
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


def extract_post_id_from_url(url: str) -> Optional[int]:
    """Extract post ID from Hacker News URL."""
    match = ID_FROM_URL_PATTERN.search(url)
    if match:
        return int(match.group(1))
    return None


def parse_iso8601(date_str: str) -> Optional[dt.datetime]:
    """Parse ISO8601-ish strings safely to aware UTC datetimes."""

    if not date_str:
        return None
    try:
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _max_date_from_items(items: List[Dict], key: str) -> Optional[str]:
    """Return max ISO date string from items[key] values (if parseable)."""

    max_dt = None
    for item in items:
        dt_obj = parse_iso8601(item.get(key, ""))
        if dt_obj and (max_dt is None or dt_obj > max_dt):
            max_dt = dt_obj
    return max_dt.isoformat() if max_dt else None


def normalize_cache_shape(data: Dict, kind: str) -> Dict:
    """Ensure cache has {items, metadata}; auto-migrate legacy list shape."""

    if isinstance(data, list):
        items = data
        metadata: Dict[str, Optional[str]] = {}
    elif isinstance(data, dict):
        items = data.get("items") if isinstance(data.get("items"), list) else []
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    else:
        items = []
        metadata = {}

    # Derive last_synced_at if missing
    if metadata.get("last_synced_at") is None:
        if kind == "comments":
            derived = _max_date_from_items(items, "date")
        elif kind == "posts":
            derived = _max_date_from_items(items, "created_at")
        else:
            derived = None
        metadata["last_synced_at"] = derived or DEFAULT_LAST_SYNC

    metadata.setdefault("schema_version", SCHEMA_VERSION)

    return {"items": items, "metadata": metadata}


def load_cache(path: Path, kind: str) -> Dict:
    """Load cache file, migrating legacy list shape in-memory if needed."""

    if not path.exists():
        return {
            "items": [],
            "metadata": {
                "last_synced_at": DEFAULT_LAST_SYNC,
                "schema_version": SCHEMA_VERSION,
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return normalize_cache_shape(raw, kind)


def migrate_cache_file(path: Path, kind: str) -> None:
    """If cache is legacy list, rewrite to {items, metadata} schema."""

    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    migrated = normalize_cache_shape(raw, kind)

    # Only rewrite if structure changed or metadata/schema_version missing
    needs_write = not isinstance(raw, dict) or "items" not in raw or "metadata" not in raw
    if isinstance(raw, dict):
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        if meta.get("schema_version") != SCHEMA_VERSION:
            needs_write = True

    if needs_write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(migrated, f, indent=2, ensure_ascii=False)


def write_cache(cache: Dict, output_path: Path) -> None:
    """Persist cache ensuring metadata defaults are present."""

    cache = normalize_cache_shape(cache, "comments" if "comments" in output_path.name else "posts")
    cache["metadata"].setdefault("last_synced_at", dt.datetime.now(dt.timezone.utc).isoformat())
    cache["metadata"]["schema_version"] = SCHEMA_VERSION

    write_json(cache, str(output_path))


def read_posts_json(json_path: str) -> List[Dict]:
    """Read posts from JSON (or legacy CSV) and return list of post data."""
    path_obj = Path(json_path)
    if path_obj.suffix.lower() == ".csv":
        with open(path_obj, "r", encoding="utf-8", newline="") as csvfile:
            return list(csv.DictReader(csvfile))

    cache = load_cache(path_obj, "posts")
    posts = cache.get("items", [])
    if not isinstance(posts, list):
        raise ValueError(f"Expected a list of posts in {json_path}")
    return posts


def fetch_comments_from_posts(
    posts_path: str,
    max_comments_per_post: Optional[int] = None,
    max_comments_total: Optional[int] = None,
    existing_cache_path: Optional[str] = None,
    refresh_cache: bool = False,
) -> Tuple[Dict, int]:
    """Fetch all comments from posts listed in JSON file and return cache dict plus new count."""
    posts = read_posts_json(posts_path)
    existing_cache = (
        {"items": [], "metadata": {"last_synced_at": None}}
        if refresh_cache
        else load_cache(Path(existing_cache_path or ""), "comments")
        if existing_cache_path
        else {"items": [], "metadata": {"last_synced_at": None}}
    )
    existing_comments = existing_cache.get("items", [])
    existing_ids = {str(c.get("id")) for c in existing_comments if c.get("id") is not None}

    last_synced_at = parse_iso8601(existing_cache.get("metadata", {}).get("last_synced_at"))
    if last_synced_at is None:
        inferred = _max_date_from_items(existing_comments, "date")
        last_synced_at = parse_iso8601(inferred) if inferred else None

    print(
        "Incremental comment fetch:",
        f"found {len(existing_comments)} existing; last_synced_at={last_synced_at.isoformat() if last_synced_at else 'none'}",
    )

    new_comments: List[Dict] = []
    global_counter = [0]

    for i, post in enumerate(posts, 1):
        post_url = post.get("hn_url", "")
        post_id = extract_post_id_from_url(post_url)

        if not post_id:
            print(f"Skipping post {i}: Could not extract ID from URL: {post_url}")
            continue

        print(
            f"Fetching comments for post {i}/{len(posts)}: {post.get('title', 'Unknown')} (ID: {post_id})"
        )

        post_comments = fetch_new_comments_algolia(post_id, post_url, last_synced_at)

        # Respect optional limits (mainly for tests)
        if max_comments_per_post is not None:
            post_comments = post_comments[:max_comments_per_post]

        for c in post_comments:
            if max_comments_total is not None and global_counter[0] >= max_comments_total:
                break

            cid = str(c.get("id")) if c.get("id") is not None else None
            if cid and cid in existing_ids:
                continue

            new_comments.append(c)
            existing_ids.add(cid or "")
            global_counter[0] += 1

        print(
            f"  Found {len(post_comments)} new top-level comments (total new so far: {len(new_comments)})"
        )

        if max_comments_total is not None and global_counter[0] >= max_comments_total:
            print("Reached max comment limit; stopping early.")
            break

    merged_comments = existing_comments + new_comments
    merged_comments.sort(
        key=lambda c: parse_iso8601(c.get("date", ""))
        or dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    )

    merged_cache = {
        "items": merged_comments,
        "metadata": {
            "last_synced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        },
    }

    return merged_cache, len(new_comments)


def write_json(data: List[Dict], output_path: str) -> None:
    """Write data to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)


def extract_job_info_with_llm(
    content: str, client: Optional[OpenAI] = None, matched_text: Optional[str] = None
) -> Dict:
    """Extract job posting information using OpenAI's LLM.

    Args:
        content: The full_content text from a match
        client: OpenAI client instance (will create if None)
        matched_text: The specific text that was matched (e.g., "Engineering Manager", "Head of Engineering")
                     This helps focus extraction on the specific role of interest.

    Returns:
        Dictionary with extracted fields:
        - company_name: str or None
        - role_name: str or None (should be the specific role that matches matched_text if provided)
        - is_remote: bool or None
        - location: str or None
        - employment_type: str or None (Full-time, Part-time, Contract, Fractional)
        - cash_compensation: str or None (original format, e.g., "$160k–$250k")
        - equity_compensation: bool or None (or str if description provided)
    """
    if OpenAI is None:
        return {
            "company_name": None,
            "role_name": None,
            "is_remote": None,
            "location": None,
            "employment_type": None,
            "cash_compensation": None,
            "equity_compensation": None,
            "extraction_error": "OpenAI library not installed",
        }

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "company_name": None,
                "role_name": None,
                "is_remote": None,
                "location": None,
                "employment_type": None,
                "cash_compensation": None,
                "equity_compensation": None,
                "extraction_error": "OPENAI_API_KEY not found in environment",
            }
        client = OpenAI(api_key=api_key)

    try:
        # Clean HTML tags for better extraction (keep text content)
        # Simple approach: remove common HTML tags
        cleaned_content = re.sub(r"<[^>]+>", " ", content)
        cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

        # Truncate if too long (keep first 4000 chars to stay within token limits)
        if len(cleaned_content) > 4000:
            cleaned_content = cleaned_content[:4000] + "..."

        # Use default fast model (not configurable via env)
        model = DEFAULT_OPENAI_MODEL

        # Build the user prompt with context about what role to focus on
        role_focus = ""
        if matched_text:
            role_focus = f"\n\nIMPORTANT: This posting was matched for the role '{matched_text}'. Please extract the SPECIFIC role that matches or contains this text (e.g., if matched_text is 'Engineering Manager', extract 'Engineering Manager' or 'Engineering Manager, Deep Learning Inference', not other unrelated roles). If the posting lists multiple roles, extract the one that matches '{matched_text}'."

        system_prompt = """You are a job posting data extractor. Extract the following information from job postings and return ONLY valid JSON with no additional text:

Fields to extract:
- company_name: The name of the company (string or null)
- role_name: The SPECIFIC role/title that matches the matched_text if provided, or the primary engineering management role mentioned (string or null). If multiple roles are listed, extract the one that matches the matched_text.
- is_remote: Whether the role is remote (boolean: true/false/null). Consider "remote-friendly", "remote-first", "hybrid" as true for remote capability.
- location: Physical location if specified, or remote location preference (string or null, e.g., "Buffalo, NY", "SF / NYC / DC", "North America preferred")
- employment_type: One of "Full-time", "Part-time", "Contract", "Fractional", or null. Normalize variations like "Full Time", "Full-Time" to "Full-time".
- cash_compensation: Salary/compensation range in original format (string or null, e.g., "$160k–$250k", "$130,000 - $250,000"). Include any salary/compensation mentioned.
- equity_compensation: Whether equity is mentioned (boolean: true/false/null) or description if provided (string). Look for terms like "equity", "stock options", "ownership".

Return JSON only, no markdown formatting, no explanations."""

        user_content = f"Extract job information from this posting:{role_focus}\n\n{cleaned_content}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,  # Deterministic output
        )

        extracted = json.loads(response.choices[0].message.content)

        # Ensure all expected fields are present
        result = {
            "company_name": extracted.get("company_name"),
            "role_name": extracted.get("role_name"),
            "is_remote": extracted.get("is_remote"),
            "location": extracted.get("location"),
            "employment_type": extracted.get("employment_type"),
            "cash_compensation": extracted.get("cash_compensation"),
            "equity_compensation": extracted.get("equity_compensation"),
        }

        return result

    except Exception as e:
        return {
            "company_name": None,
            "role_name": None,
            "is_remote": None,
            "location": None,
            "employment_type": None,
            "cash_compensation": None,
            "equity_compensation": None,
            "extraction_error": str(e),
        }


def compile_patterns_from_profile(profile_path: Path) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns for role search from a YAML profile."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    patterns = []
    for idx, entry in enumerate(data.get("patterns", []), 1):
        regex = None
        pattern_name = None

        if isinstance(entry, str):
            regex = entry
        elif isinstance(entry, dict):
            regex = entry.get("regex")
            pattern_name = entry.get("name")
        else:
            continue

        if not regex:
            continue

        try:
            compiled = re.compile(regex, re.IGNORECASE)
            patterns.append((compiled, pattern_name or f"pattern_{idx}"))
        except re.error as e:
            display_name = pattern_name or f"pattern_{idx}"
            print(f"Skipping pattern '{display_name}' due to regex error: {e}")

    return patterns


def search_engineering_management_roles(
    comments_path: str,
    extract_with_llm: bool = True,
    profile_path: Optional[str] = None,
    max_matches: Optional[int] = None,
) -> List[Dict]:
    """Search comments for engineering management role postings.

    Args:
        comments_path: Path to JSON file containing comments
        extract_with_llm: Whether to extract structured data using LLM (default: True)
        profile_path: Path to YAML profile defining regex patterns (default: engineering management profile)

    Returns:
        List of match dictionaries with comment info and matched text
    """
    # Load comments
    print(f"Loading comments from {comments_path}...")
    comments_cache = load_cache(Path(comments_path), "comments")
    comments = comments_cache.get("items", [])

    print(f"Loaded {len(comments)} comments")

    # Compile patterns
    profile_file = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
    patterns = compile_patterns_from_profile(profile_file)
    print(f"Using {len(patterns)} search patterns from profile: {profile_file.name}")

    # Initialize OpenAI client if extraction is enabled
    client = None
    if extract_with_llm:
        if OpenAI is None:
            print("Warning: OpenAI library not installed. Skipping LLM extraction.")
            print("Install with: pip install openai python-dotenv")
            extract_with_llm = False
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(
                    "Warning: OPENAI_API_KEY not found in environment. Skipping LLM extraction."
                )
                print("Create a .env file with: OPENAI_API_KEY=your_key_here")
                extract_with_llm = False
            else:
                client = OpenAI(api_key=api_key)
                print(f"LLM extraction enabled (using {DEFAULT_OPENAI_MODEL})")

    matches = []

    # Search through each comment
    stop_search = False
    for idx, comment in enumerate(comments):
        if (idx + 1) % 1000 == 0:
            print(
                f"  Processed {idx + 1}/{len(comments)} comments, found {len(matches)} matches so far..."
            )

        if max_matches is not None and len(matches) >= max_matches:
            stop_search = True
            break

        content = comment.get("content", "")
        if not content:
            continue

        # Decode HTML entities
        decoded_content = html.unescape(content)

        # Try each pattern
        for pattern, pattern_name in patterns:
            for match in pattern.finditer(decoded_content):
                # Extract surrounding context (100 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(decoded_content), match.end() + 100)
                context = decoded_content[start:end]

                match_dict = {
                    "comment_index": idx,
                    "comment_id": comment.get("id"),
                    "post_url": comment.get("post_url", ""),
                    "commenter": comment.get("commenter", ""),
                    "date": comment.get("date", ""),
                    "matched_text": match.group(0),
                    "pattern_name": pattern_name,
                    "context": context,
                    "full_content": decoded_content,  # Include full content for review
                }

                # Extract structured data using LLM if enabled
                if extract_with_llm and client:
                    comment_id = match_dict.get("comment_id")
                    if len(matches) % 10 == 0 and len(matches) > 0:
                        print(
                            f"  Extracting data for match {len(matches)} (comment {comment_id or 'n/a'})..."
                        )
                    extracted = extract_job_info_with_llm(
                        decoded_content, client, matched_text=match.group(0)
                    )
                    match_dict["extracted"] = extracted
                    # Small delay to avoid rate limits (can be removed if using batch processing)
                    time.sleep(0.1)

                matches.append(match_dict)
                # Only record first match per pattern per comment to avoid duplicates
                break

            if max_matches is not None and len(matches) >= max_matches:
                stop_search = True
                break

        if stop_search:
            break

    print(f"\nFound {len(matches)} total matches")
    if extract_with_llm:
        successful_extractions = sum(
            1
            for m in matches
            if m.get("extracted") and not m.get("extracted", {}).get("extraction_error")
        )
        print(
            f"Successfully extracted data for {successful_extractions}/{len(matches)} matches"
        )
    return matches


def extract_from_matches(input_path: str, output_path: str, reextract: bool = False) -> None:
    """Read matches from JSON file and add extracted data to each match.

    Args:
        input_path: Path to input JSON file with matches
        output_path: Path to write output JSON file with extracted data
        reextract: Force rerun extraction even if match already has extracted data
    """
    print(f"Loading matches from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")
    already_extracted = sum(
        1
        for m in matches
        if m.get("extracted") and not m.get("extracted", {}).get("extraction_error")
    )
    pending = len(matches) - already_extracted
    print(
        f"Extraction status: {already_extracted} already extracted, {pending} pending (use --reextract to force)."
    )

    # Initialize OpenAI client
    if OpenAI is None:
        print("Error: OpenAI library not installed. Install with: pip install openai")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Create a .env file with: OPENAI_API_KEY=your_key_here")
        return

    client = OpenAI(api_key=api_key)
    print(f"Using model: {DEFAULT_OPENAI_MODEL}")
    print(f"Extracting data for {len(matches)} matches...\n")

    # Extract data for each match
    for idx, match in enumerate(matches, 1):
        if not reextract:
            existing = match.get("extracted")
            if existing and not existing.get("extraction_error"):
                print(
                    f"Match {idx}/{len(matches)} already extracted; skipping (use --reextract to force)."
                )
                continue

        full_content = match.get("full_content", "")
        if not full_content:
            print(f"Match {idx}/{len(matches)}: No full_content, skipping")
            continue

        comment_id = match.get("comment_id")
        print(
            f"Extracting data for match {idx}/{len(matches)} (comment {comment_id or 'n/a'})..."
        )
        matched_text = match.get("matched_text", None)
        extracted = extract_job_info_with_llm(
            full_content, client, matched_text=matched_text
        )
        match["extracted"] = extracted

        # Show what was extracted
        if extracted.get("extraction_error"):
            print(f"  ⚠️  Error: {extracted['extraction_error']}")
        else:
            company = extracted.get("company_name") or "N/A"
            role = extracted.get("role_name") or "N/A"
            remote = (
                "Yes"
                if extracted.get("is_remote")
                else "No" if extracted.get("is_remote") is False else "N/A"
            )
            comp = extracted.get("cash_compensation") or "N/A"
            print(
                f"  ✓ Company: {company}, Role: {role}, Remote: {remote}, Comp: {comp}"
            )

        # Small delay to avoid rate limits
        time.sleep(0.1)

    # Write results
    print(f"\nWriting results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    # Summary
    successful = sum(
        1
        for m in matches
        if m.get("extracted") and not m.get("extracted", {}).get("extraction_error")
    )
    print(
        f"\n✅ Complete! Successfully extracted data for {successful}/{len(matches)} matches"
    )
    print(f"Results written to {output_path}")


def generate_html_report(
    input_path: str, output_path: str, open_browser: bool = True
) -> None:
    """Generate a beautiful self-contained HTML report from matches with extraction.

    Args:
        input_path: Path to input JSON file with matches and extracted data
        output_path: Path to write output HTML file
        open_browser: Whether to open the report in the default browser
    """
    print(f"Loading matches from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")
    print(f"Generating HTML report...")

    # Format date for display
    def format_date(date_str: Optional[str]) -> str:
        if not date_str:
            return ""
        try:
            dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt_obj.strftime("%Y-%m-%d")
        except Exception:
            return date_str

    rows: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        extracted = match.get("extracted", {}) or {}
        is_remote_val = extracted.get("is_remote")
        full_content_raw = match.get("full_content", "") or ""
        rows.append(
            {
                "row_id": f"row-{idx}",
                "company": (extracted.get("company_name") or "").strip(),
                "role": (extracted.get("role_name") or "").strip(),
                "location": (extracted.get("location") or "").strip(),
                "remote": (
                    "Yes" if is_remote_val else "No" if is_remote_val is False else "—"
                ),
                "remote_class": (
                    "remote-yes"
                    if is_remote_val
                    else "remote-no" if is_remote_val is False else ""
                ),
                "employment": (extracted.get("employment_type") or "").strip(),
                "cash_comp": (extracted.get("cash_compensation") or "").strip(),
                "equity": (
                    "Yes"
                    if extracted.get("equity_compensation")
                    else "No" if extracted.get("equity_compensation") is False else "—"
                ),
                "commenter": (match.get("commenter", "") or "").strip(),
                "date": format_date(match.get("date", "") or ""),
                "post_url": match.get("post_url", "") or "",
                "matched_text": match.get("matched_text", "") or "",
                "full_content": full_content_raw,
                "full_content_html": sanitize_full_content(full_content_raw),
            }
        )

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    html_content = template.render(rows=rows, total_matches=len(rows))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report generated: {output_path}")
    print(f"   Open in your browser to view {len(matches)} matches")

    if open_browser:
        try:
            webbrowser.open(output_path.resolve().as_uri())
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not open browser automatically: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    description="Fetch 'Ask HN: Who is hiring?' threads and save them to JSON, or fetch comments from posts, or search for roles using a profile, or extract structured data from matches, or generate an HTML report."
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Mode: fetch comments from posts in JSON file instead of fetching post URLs.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "Mode: search comments for roles using regex patterns defined in a profile (default profile: engineering_management)."
        ),
    )
    parser.add_argument(
        "--profile",
        help="Path to YAML profile defining role search patterns (default: profiles/engineering_management.yaml). Used for search (and to set per-profile output defaults).",
    )
    parser.add_argument(
        "--extract-from-matches",
        action="store_true",
        help="Mode: extract structured data from existing matches JSON file using LLM.",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        help="Mode: generate a beautiful self-contained HTML report from matches with extraction data.",
    )
    parser.add_argument(
        "--no-open-report",
        action="store_true",
        help="Skip automatically opening the HTML report after generation (useful in CI/tests). Only used with --generate-html.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip LLM-based extraction of structured data (faster, but no extracted fields). Only used with --search.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        help="Limit number of posts to fetch (for tests)",
    )
    parser.add_argument(
        "--post-id",
        type=int,
        help="Fetch a specific 'Who is hiring?' post ID instead of searching",
    )
    parser.add_argument(
        "--comments-per-post",
        type=int,
        help="Limit number of comments fetched per post (for tests)",
    )
    parser.add_argument(
        "--max-comments-total",
        type=int,
        help="Limit total number of comments fetched across posts (for tests)",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        help="Limit number of matches processed (for tests)",
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Force rerunning LLM extraction even if matches already contain extracted data.",
    )
    parser.add_argument(
        "--input",
        help="Input file (JSON for all modes; defaults are out/posts.json, out/comments.json, out/matches.json, and out/matches_with_extraction.json respectively).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="How many months back to search (approximate, default: 6). Only used in default mode.",
    )
    parser.add_argument(
        "--output",
        help="Path to write output (defaults depend on mode: posts.json for posts, out/comments.json for comments, out/<profile>/matches.json for search, out/<profile>/matches_with_extraction.json for extraction, out/<profile>/report.html for HTML).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force regeneration even if posts.json or comments.json already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Hard fail fast if OPENAI_API_KEY is not set so behavior is predictable.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found in environment. "
            "Please set it in your shell or a .env file before running who_is_hiring.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-migrate default caches to new schema when present
    migrate_cache_file(DEFAULT_POSTS_PATH, "posts")
    migrate_cache_file(DEFAULT_COMMENTS_PATH, "comments")

    slug = profile_slug(args.profile)
    default_matches_path = OUT_DIR / slug / "matches.json"
    default_matches_with_extraction_path = (
        OUT_DIR / slug / "matches_with_extraction.json"
    )
    default_report_path = OUT_DIR / slug / "report.html"

    if args.generate_html:
        # Mode 5: Generate HTML report
        input_path = args.input or str(default_matches_with_extraction_path)
        output_path = args.output or str(default_report_path)

        generate_html_report(
            input_path, output_path, open_browser=not args.no_open_report
        )
    elif args.extract_from_matches:
        # Mode 4: Extract structured data from existing matches
        input_path = args.input or str(default_matches_path)
        output_path = args.output or str(default_matches_with_extraction_path)

        # Safety check: don't overwrite comments.json
        if Path(output_path).name == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches_with_extraction.json instead."
            )
            output_path = str(default_matches_with_extraction_path)

        extract_from_matches(input_path, output_path, reextract=args.reextract)
    elif args.search:
        # Mode 3: Search comments for roles using a profile
        input_path = args.input or str(DEFAULT_COMMENTS_PATH)
        output_path = args.output or str(default_matches_path)
        profile_arg = args.profile

        if profile_arg:
            profile_path_obj = Path(profile_arg)
            if not profile_path_obj.exists():
                print(f"Error: Profile file not found: {profile_arg}")
                sys.exit(1)

        # Safety check: don't overwrite comments.json
        if Path(output_path).name == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches.json instead."
            )
            output_path = str(DEFAULT_MATCHES_PATH)

        matches = search_engineering_management_roles(
            input_path,
            extract_with_llm=not args.no_extract,
            profile_path=profile_arg,
            max_matches=args.max_matches,
        )
        write_json(matches, output_path)
        print(f"\nWrote {len(matches)} matches to {output_path}")
    elif args.fetch_comments:
        # Mode 2: Fetch comments from posts
        input_path = args.input or str(DEFAULT_POSTS_PATH)
        output_path = args.output or str(DEFAULT_COMMENTS_PATH)
        output_path_obj = Path(output_path)

        # Migrate any legacy cache shape before reading
        migrate_cache_file(Path(input_path), "posts")
        if output_path_obj.exists():
            migrate_cache_file(output_path_obj, "comments")

        if not Path(input_path).exists():
            print(f"Error: Posts file not found: {input_path}")
            return

        comments_cache, new_count = fetch_comments_from_posts(
            input_path,
            max_comments_per_post=args.comments_per_post,
            max_comments_total=args.max_comments_total,
            existing_cache_path=output_path if output_path_obj.exists() else None,
            refresh_cache=args.refresh_cache,
        )
        write_cache(comments_cache, output_path_obj)
        total = len(comments_cache.get("items", []))
        print(
            f"\nSaved {total} comments to {output_path} (added {new_count}, unchanged {total - new_count})."
        )
    else:
        # Mode 1: Fetch post URLs (original functionality)
        output_path = args.output or str(DEFAULT_POSTS_OUTPUT)
        output_path_obj = Path(output_path)

        if args.post_id:
            post_item = fetch_hn_item(args.post_id)
            if not post_item:
                print(f"Error: Could not fetch post with ID {args.post_id}")
                return

            title = (post_item.get("title") or "").strip()
            if not TITLE_PATTERN.match(title):
                print(
                    f"Warning: Post ID {args.post_id} title does not match 'Who is hiring?' pattern: {title}"
                )

            created_at = dt.datetime.fromtimestamp(
                post_item.get("time", 0), tz=dt.timezone.utc
            ).isoformat()
            threads = [
                {
                    "id": str(post_item.get("id", "")),
                    "title": title,
                    "author": post_item.get("by", ""),
                    "created_at": created_at,
                    "hn_url": f"https://news.ycombinator.com/item?id={post_item.get('id', '')}",
                    "points": post_item.get("score"),
                    "num_comments": len(post_item.get("kids", []) or []),
                }
            ]
        else:
            since_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
                days=31 * args.months
            )
            threads = fetch_who_is_hiring_threads(since_date, max_posts=args.max_posts)
        write_posts_json(threads, output_path)
        print(f"Wrote {len(threads)} threads to {output_path}")


if __name__ == "__main__":
    main()
