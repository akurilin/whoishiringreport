"""Fetch recent 'Ask HN: Who is hiring?' threads and save them to CSV.

Run with: python who_is_hiring.py --months 24 --output who_is_hiring_posts.csv
Or fetch comments: python who_is_hiring.py --fetch-comments --input posts.csv --output out/comments.json
Or search for engineering management roles: python who_is_hiring.py --search-eng-management --input out/comments.json --output out/matches.json
Or extract from matches: python who_is_hiring.py --extract-from-matches --input out/matches.json --output out/matches_with_extraction.json
Or generate HTML report: python who_is_hiring.py --generate-html --input out/matches_with_extraction.json --output out/report.html
"""

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
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
DEFAULT_POSTS_CSV = BASE_DIR / "posts.csv"
DEFAULT_POSTS_OUTPUT = BASE_DIR / "who_is_hiring_posts.csv"
DEFAULT_COMMENTS_PATH = OUT_DIR / "comments.json"
DEFAULT_MATCHES_PATH = OUT_DIR / "matches.json"
DEFAULT_MATCHES_WITH_EXTRACTION_PATH = OUT_DIR / "matches_with_extraction.json"
DEFAULT_REPORT_PATH = OUT_DIR / "report.html"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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


def fetch_comments_from_posts(
    csv_path: str,
    max_comments_per_post: Optional[int] = None,
    max_comments_total: Optional[int] = None,
) -> List[Dict]:
    """Fetch all comments from posts listed in CSV file."""
    posts = read_posts_csv(csv_path)
    all_comments = []
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
        post_comments = []
        fetch_all_comments(
            post_id,
            post_url,
            post_comments,
            max_comments_per_post=max_comments_per_post,
            global_counter=global_counter,
            max_comments_total=max_comments_total,
        )
        all_comments.extend(post_comments)
        print(
            f"  Found {len(post_comments)} comments (total so far: {len(all_comments)})"
        )

        if (
            max_comments_total is not None
            and global_counter[0] >= max_comments_total
        ):
            print("Reached max comment limit; stopping early.")
            break

    return all_comments


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
    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

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
                    if len(matches) % 10 == 0 and len(matches) > 0:
                        print(f"  Extracting data for match {len(matches)}...")
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


def extract_from_matches(input_path: str, output_path: str) -> None:
    """Read matches from JSON file and add extracted data to each match.

    Args:
        input_path: Path to input JSON file with matches
        output_path: Path to write output JSON file with extracted data
    """
    print(f"Loading matches from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")

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
        full_content = match.get("full_content", "")
        if not full_content:
            print(f"Match {idx}/{len(matches)}: No full_content, skipping")
            continue

        print(f"Extracting data for match {idx}/{len(matches)}...")
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
        rows.append(
            {
                "row_id": f"row-{idx}",
                "company": (extracted.get("company_name") or "").strip(),
                "role": (extracted.get("role_name") or "").strip(),
                "location": (extracted.get("location") or "").strip(),
                "remote": "Yes"
                if is_remote_val
                else "No"
                if is_remote_val is False
                else "—",
                "remote_class": "remote-yes"
                if is_remote_val
                else "remote-no"
                if is_remote_val is False
                else "",
                "employment": (extracted.get("employment_type") or "").strip(),
                "cash_comp": (extracted.get("cash_compensation") or "").strip(),
                "equity": "Yes"
                if extracted.get("equity_compensation")
                else "No"
                if extracted.get("equity_compensation") is False
                else "—",
                "commenter": (match.get("commenter", "") or "").strip(),
                "date": format_date(match.get("date", "") or ""),
                "post_url": match.get("post_url", "") or "",
                "matched_text": match.get("matched_text", "") or "",
                "full_content": match.get("full_content", "") or "",
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
        description="Fetch 'Ask HN: Who is hiring?' threads and save them to CSV, or fetch comments from posts, or search for engineering management roles, or extract structured data from matches, or generate an HTML report."
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Mode: fetch comments from posts in CSV file instead of fetching post URLs.",
    )
    parser.add_argument(
        "--search-eng-management",
        action="store_true",
        help="Mode: search comments for engineering management roles (Head of Eng, VP Eng, Director of Engineering, etc.).",
    )
    parser.add_argument(
        "--profile",
        help="Path to YAML profile defining role search patterns (default: profiles/engineering_management.yaml). Only used with --search-eng-management.",
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
        help="Skip LLM-based extraction of structured data (faster, but no extracted fields). Only used with --search-eng-management.",
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
        "--input",
        help="Input file (CSV for --fetch-comments mode, JSON for --search-eng-management, --extract-from-matches, and --generate-html modes; defaults are posts.csv, out/comments.json, out/matches.json, and out/matches_with_extraction.json respectively).",
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
        help="Path to write output (default: who_is_hiring_posts.csv for posts, out/comments.json for comments, out/matches.json for search, out/matches_with_extraction.json for extraction, out/report.html for HTML).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if args.generate_html:
        # Mode 5: Generate HTML report
        input_path = input_path or str(DEFAULT_MATCHES_WITH_EXTRACTION_PATH)
        if not output_path.endswith(".html"):
            # Default to out/report.html if not specified
            if output_path == "who_is_hiring_posts.csv":
                output_path = str(DEFAULT_REPORT_PATH)

        generate_html_report(
            input_path, output_path, open_browser=not args.no_open_report
        )
    elif args.extract_from_matches:
        # Mode 4: Extract structured data from existing matches
        input_path = input_path or str(DEFAULT_MATCHES_PATH)
        if not output_path.endswith(".json"):
            # Default to matches_with_extraction.json if not specified
            if output_path == "who_is_hiring_posts.csv":
                output_path = str(DEFAULT_MATCHES_WITH_EXTRACTION_PATH)

        # Safety check: don't overwrite comments.json
        if output_path == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches_with_extraction.json instead."
            )
            output_path = str(DEFAULT_MATCHES_WITH_EXTRACTION_PATH)

        extract_from_matches(input_path, output_path)
    elif args.search_eng_management:
        # Mode 3: Search for engineering management roles
        input_path = input_path or str(DEFAULT_COMMENTS_PATH)
        if not output_path.endswith(".json"):
            # Default to matches.json if not specified
            if output_path == "who_is_hiring_posts.csv":
                output_path = str(DEFAULT_MATCHES_PATH)

        # Safety check: don't overwrite comments.json
        if output_path == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches.json instead."
            )
            output_path = str(DEFAULT_MATCHES_PATH)

        matches = search_engineering_management_roles(
            input_path,
            extract_with_llm=not args.no_extract,
            profile_path=args.profile,
            max_matches=args.max_matches,
        )
        write_json(matches, output_path)
        print(f"\nWrote {len(matches)} matches to {output_path}")
    elif args.fetch_comments:
        # Mode 2: Fetch comments from posts in CSV
        input_path = input_path or str(DEFAULT_POSTS_CSV)
        if not output_path.endswith(".json"):
            # Default to .json if not specified
            if output_path == "who_is_hiring_posts.csv":
                output_path = str(DEFAULT_COMMENTS_PATH)

        comments = fetch_comments_from_posts(
            input_path,
            max_comments_per_post=args.comments_per_post,
            max_comments_total=args.max_comments_total,
        )
        write_json(comments, output_path)
        print(f"\nWrote {len(comments)} comments to {output_path}")
    else:
        # Mode 1: Fetch post URLs (original functionality)
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
            threads = fetch_who_is_hiring_threads(
                since_date, max_posts=args.max_posts
            )
        output_path = output_path or str(DEFAULT_POSTS_OUTPUT)
        write_csv(threads, output_path)
        print(f"Wrote {len(threads)} threads to {output_path}")


if __name__ == "__main__":
    main()
