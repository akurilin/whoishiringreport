"""Fetch recent 'Ask HN: Who is hiring?' threads and save them to CSV.

Run with: python who_is_hiring.py --months 24 --output who_is_hiring_posts.csv
Or fetch comments: python who_is_hiring.py --fetch-comments --input posts.csv --output comments.json
Or search for engineering management roles: python who_is_hiring.py --search-eng-management --input comments.json --output matches.json
Or extract from matches: python who_is_hiring.py --extract-from-matches --input matches.json --output matches_with_extraction.json
Or generate HTML report: python who_is_hiring.py --generate-html --input matches_with_extraction.json --output report.html
"""

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

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

        # Use model from environment or default to gpt-4o (fast, non-thinking model)
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

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


def compile_engineering_management_patterns() -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns for matching engineering management roles.

    Returns a list of (pattern, pattern_name) tuples.
    """
    patterns = []

    # Pattern Group A: [Title] of [Engineering]
    # e.g., "Head of Engineering", "VP of Software Development"
    pattern_a = re.compile(
        r"\b(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr)\s+of\s+(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology|programming)\b",
        re.IGNORECASE,
    )
    patterns.append((pattern_a, "title_of_engineering"))

    # Pattern Group B: [Engineering] [Title]
    # e.g., "Engineering Director", "Software Engineering Manager"
    # Note: Excludes "lead|leader" to avoid IC roles like "Engineering Lead"
    pattern_b = re.compile(
        r"\b(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology|programming)\s+(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr)\b",
        re.IGNORECASE,
    )
    patterns.append((pattern_b, "engineering_title"))

    # Pattern Group C: [Title], [Engineering]
    # e.g., "Director, Engineering"
    pattern_c = re.compile(
        r"\b(head|director|dir|der|vp|vpe|v\.?p\.?|vice\s+president|manager|mgr),\s*(eng|eng\'?g|engr|engineering|software\s+engineering|software\s+development|dev|development|tech|technology)\b",
        re.IGNORECASE,
    )
    patterns.append((pattern_c, "title_comma_engineering"))

    # Pattern Group D: Abbreviated forms
    # e.g., "VP Eng", "Dir of ENG", "DER"
    pattern_d = re.compile(
        r"\b(head|dir|der|vp|vpe)\s+(of\s+)?(eng|eng\'?g|engr)\b", re.IGNORECASE
    )
    patterns.append((pattern_d, "abbreviated_form"))

    # Pattern Group E: Senior/Principal with management
    # e.g., "Senior Engineering Manager", "Principal Software Director"
    # Note: Excludes "lead" to avoid IC roles like "Senior Engineering Lead"
    pattern_e = re.compile(
        r"\b(senior|principal)\s+(engineering|software|dev|development)\s+(manager|director|head|vp|vpe)\b",
        re.IGNORECASE,
    )
    patterns.append((pattern_e, "senior_principal_management"))

    return patterns


def search_engineering_management_roles(
    comments_path: str, extract_with_llm: bool = True
) -> List[Dict]:
    """Search comments for engineering management role postings.

    Args:
        comments_path: Path to JSON file containing comments
        extract_with_llm: Whether to extract structured data using LLM (default: True)

    Returns:
        List of match dictionaries with comment info and matched text
    """
    # Load comments
    print(f"Loading comments from {comments_path}...")
    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    print(f"Loaded {len(comments)} comments")

    # Compile patterns
    patterns = compile_engineering_management_patterns()
    print(f"Using {len(patterns)} search patterns")

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
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
                print(f"LLM extraction enabled (using {model})")

    matches = []

    # Search through each comment
    for idx, comment in enumerate(comments):
        if (idx + 1) % 1000 == 0:
            print(
                f"  Processed {idx + 1}/{len(comments)} comments, found {len(matches)} matches so far..."
            )

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
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"Using model: {model}")
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


def generate_html_report(input_path: str, output_path: str) -> None:
    """Generate a beautiful self-contained HTML report from matches with extraction.

    Args:
        input_path: Path to input JSON file with matches and extracted data
        output_path: Path to write output HTML file
    """
    print(f"Loading matches from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")
    print(f"Generating HTML report...")

    # Escape HTML entities for safe display
    def escape_html(text):
        if text is None:
            return ""
        return html.escape(str(text))

    # Format date for display
    def format_date(date_str):
        if not date_str:
            return ""
        try:
            dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt_obj.strftime("%Y-%m-%d")
        except:
            return date_str

    # Generate table rows
    table_rows = []
    for idx, match in enumerate(matches):
        extracted = match.get("extracted", {})
        company = escape_html(extracted.get("company_name") or "")
        role = escape_html(extracted.get("role_name") or "")
        location = escape_html(extracted.get("location") or "")
        is_remote_val = extracted.get("is_remote")
        remote = "Yes" if is_remote_val else "No" if is_remote_val is False else "—"
        remote_class = (
            "remote-yes"
            if is_remote_val
            else "remote-no" if is_remote_val is False else ""
        )
        employment = escape_html(extracted.get("employment_type") or "")
        cash_comp = escape_html(extracted.get("cash_compensation") or "")
        equity = (
            "Yes"
            if extracted.get("equity_compensation")
            else "No" if extracted.get("equity_compensation") is False else "—"
        )
        commenter = escape_html(match.get("commenter", ""))
        date = format_date(match.get("date", ""))
        post_url = escape_html(match.get("post_url", ""))
        full_content = escape_html(match.get("full_content", ""))
        matched_text = escape_html(match.get("matched_text", ""))

        # Create expandable row with full content
        row_id = f"row-{idx}"
        table_rows.append(
            f"""
        <tr id="{row_id}" class="data-row" data-expanded="false">
            <td class="company-cell">{company}</td>
            <td class="role-cell">{role}</td>
            <td class="location-cell">{location}</td>
            <td class="remote-cell {remote_class}">{remote}</td>
            <td class="employment-cell">{employment}</td>
            <td class="cash-comp-cell">{cash_comp}</td>
            <td class="equity-cell">{equity}</td>
            <td class="commenter-cell">{commenter}</td>
            <td class="date-cell">{date}</td>
            <td class="actions-cell">
                <button class="expand-btn" onclick="toggleRow('{row_id}')">View</button>
                <a href="{post_url}" target="_blank" class="link-btn">HN</a>
            </td>
        </tr>
        <tr id="{row_id}-detail" class="detail-row" style="display: none;">
            <td colspan="10" class="detail-content">
                <div class="detail-section">
                    <h4>Matched Text:</h4>
                    <p class="matched-text">{matched_text}</p>
                </div>
                <div class="detail-section">
                    <h4>Full Content:</h4>
                    <div class="full-content">{full_content.replace(chr(10), '<br>')}</div>
                </div>
            </td>
        </tr>
        """
        )

    rows_html = "\n".join(table_rows)

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Engineering Management Job Matches</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .controls {{
            padding: 25px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .search-filter {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr auto;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
        }}
        
        .filter-group label {{
            font-weight: 600;
            margin-bottom: 5px;
            color: #495057;
            font-size: 0.9em;
        }}
        
        .filter-group input,
        .filter-group select {{
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }}
        
        .filter-group input:focus,
        .filter-group select:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #dee2e6;
        }}
        
        .stat-item {{
            flex: 1;
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .table-wrapper {{
            overflow-x: auto;
            max-height: calc(100vh - 400px);
            overflow-y: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}
        
        thead {{
            position: sticky;
            top: 0;
            background: #667eea;
            color: white;
            z-index: 10;
        }}
        
        th {{
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            transition: background-color 0.2s;
        }}
        
        th:hover {{
            background: #5568d3;
        }}
        
        th.sortable::after {{
            content: " ↕";
            opacity: 0.5;
            font-size: 0.8em;
        }}
        
        th.sort-asc::after {{
            content: " ↑";
            opacity: 1;
        }}
        
        th.sort-desc::after {{
            content: " ↓";
            opacity: 1;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #e9ecef;
            transition: background-color 0.2s;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        tbody tr.detail-row {{
            background: #f8f9fa;
        }}
        
        td {{
            padding: 12px;
            vertical-align: top;
        }}
        
        .company-cell {{
            font-weight: 600;
            color: #212529;
        }}
        
        .role-cell {{
            color: #495057;
        }}
        
        .location-cell {{
            color: #6c757d;
        }}
        
        .remote-cell {{
            text-align: center;
        }}
        
        .remote-yes {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .remote-no {{
            color: #6c757d;
        }}
        
        .cash-comp-cell {{
            color: #28a745;
            font-weight: 500;
        }}
        
        .actions-cell {{
            display: flex;
            gap: 8px;
        }}
        
        .expand-btn,
        .link-btn {{
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            text-decoration: none;
            transition: all 0.2s;
        }}
        
        .expand-btn {{
            background: #667eea;
            color: white;
        }}
        
        .expand-btn:hover {{
            background: #5568d3;
        }}
        
        .link-btn {{
            background: #6c757d;
            color: white;
            display: inline-block;
        }}

        .reset-btn {{
            background: #f8f9fa;
            color: #343a40;
            border: 2px solid #dee2e6;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .reset-btn:hover {{
            background: #e9ecef;
        }}

        .filter-actions {{
            display: flex;
            align-items: flex-end;
            justify-content: flex-end;
        }}
        
        .link-btn:hover {{
            background: #5a6268;
        }}
        
        .detail-content {{
            padding: 20px;
            background: white;
            border-left: 4px solid #667eea;
        }}
        
        .detail-section {{
            margin-bottom: 20px;
        }}
        
        .detail-section h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .matched-text {{
            font-weight: 600;
            color: #495057;
            padding: 8px;
            background: #e9ecef;
            border-radius: 4px;
            display: inline-block;
        }}
        
        .full-content {{
            line-height: 1.6;
            color: #212529;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
            font-size: 1.2em;
        }}
        
        @media (max-width: 1200px) {{
            .search-filter {{
                grid-template-columns: 1fr 1fr 1fr;
            }}
        }}
        
        @media (max-width: 768px) {{
            .search-filter {{
                grid-template-columns: 1fr;
            }}
            
            .table-wrapper {{
                font-size: 0.85em;
            }}
            
            th, td {{
                padding: 8px 6px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Engineering Management Job Matches</h1>
            <p>Found {len(matches)} matches from Hacker News "Who is hiring?" threads</p>
        </div>
        
        <div class="controls">
            <div class="search-filter">
                <div class="filter-group">
                    <label for="search">Search (Company, Role, Location...)</label>
                    <input type="text" id="search" placeholder="Type to search..." oninput="filterTable()">
                </div>
                <div class="filter-group">
                    <label for="remote-filter">Remote</label>
                    <select id="remote-filter" onchange="filterTable()">
                        <option value="">All</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="location-filter">Location</label>
                    <input type="text" id="location-filter" placeholder="e.g., San Francisco" oninput="filterTable()">
                </div>
                <div class="filter-group">
                    <label for="company-filter">Company</label>
                    <input type="text" id="company-filter" placeholder="Company name..." oninput="filterTable()">
                </div>
                <div class="filter-actions">
                    <button type="button" class="reset-btn" onclick="clearFilters()">Clear filters</button>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="total-count">{len(matches)}</div>
                    <div class="stat-label">Total Matches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="visible-count">{len(matches)}</div>
                    <div class="stat-label">Visible</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="remote-count">—</div>
                    <div class="stat-label">Remote Available</div>
                </div>
            </div>
        </div>
        
        <div class="table-wrapper">
            <table id="matches-table">
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(0)">Company</th>
                        <th class="sortable" onclick="sortTable(1)">Role</th>
                        <th class="sortable" onclick="sortTable(2)">Location</th>
                        <th class="sortable" onclick="sortTable(3)">Remote</th>
                        <th class="sortable" onclick="sortTable(4)">Type</th>
                        <th class="sortable" onclick="sortTable(5)">Compensation</th>
                        <th class="sortable" onclick="sortTable(6)">Equity</th>
                        <th class="sortable" onclick="sortTable(7)">Commenter</th>
                        <th class="sortable" onclick="sortTable(8)">Date</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="table-body">
                    {rows_html}
                </tbody>
            </table>
            <div id="no-results" class="no-results" style="display: none;">
                No matches found. Try adjusting your filters.
            </div>
        </div>
    </div>
    
    <script>
        let allRows = [];
        let currentSort = {{ column: -1, direction: 'asc' }};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            const rows = document.querySelectorAll('.data-row');
            allRows = Array.from(rows);
            updateStats();
        }});

        function clearFilters() {{
            document.getElementById('search').value = '';
            document.getElementById('remote-filter').value = '';
            document.getElementById('location-filter').value = '';
            document.getElementById('company-filter').value = '';
            filterTable();
        }}
        
        function filterTable() {{
            const search = document.getElementById('search').value.toLowerCase();
            const remoteFilter = document.getElementById('remote-filter').value;
            const locationFilter = document.getElementById('location-filter').value.toLowerCase();
            const companyFilter = document.getElementById('company-filter').value.toLowerCase();
            
            let visibleCount = 0;
            let remoteCount = 0;
            
            allRows.forEach((row, idx) => {{
                const cells = row.querySelectorAll('td');
                const company = cells[0].textContent.toLowerCase();
                const role = cells[1].textContent.toLowerCase();
                const location = cells[2].textContent.toLowerCase();
                const remote = cells[3].textContent.trim();
                const detailRow = document.getElementById(row.id + '-detail');
                const isExpanded = row.dataset.expanded === 'true';
                
                // Check filters
                const matchesSearch = !search || 
                    company.includes(search) || 
                    role.includes(search) || 
                    location.includes(search);
                const matchesRemote = !remoteFilter || remote === remoteFilter;
                const matchesLocation = !locationFilter || location.includes(locationFilter);
                const matchesCompany = !companyFilter || company.includes(companyFilter);
                
                if (matchesSearch && matchesRemote && matchesLocation && matchesCompany) {{
                    row.style.display = '';
                    if (detailRow) detailRow.style.display = isExpanded ? '' : 'none';
                    visibleCount++;
                    if (remote === 'Yes') remoteCount++;
                }} else {{
                    row.style.display = 'none';
                    if (detailRow) detailRow.style.display = 'none';
                }}
            }});
            
            document.getElementById('visible-count').textContent = visibleCount;
            document.getElementById('remote-count').textContent = remoteCount || '—';
            document.getElementById('no-results').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        function sortTable(column) {{
            const tbody = document.getElementById('table-body');
            const rows = Array.from(tbody.querySelectorAll('.data-row'));
            const headers = document.querySelectorAll('th');
            
            // Reset header classes
            headers.forEach(h => {{
                h.classList.remove('sort-asc', 'sort-desc');
            }});
            
            // Determine sort direction
            if (currentSort.column === column) {{
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSort.column = column;
                currentSort.direction = 'asc';
            }}
            
            // Update header
            headers[column].classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Sort rows
            rows.sort((a, b) => {{
                const aText = a.querySelectorAll('td')[column].textContent.trim();
                const bText = b.querySelectorAll('td')[column].textContent.trim();
                
                // Try numeric comparison
                const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ''));
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return currentSort.direction === 'asc' ? aNum - bNum : bNum - aNum;
                }}
                
                // String comparison
                const comparison = aText.localeCompare(bText, undefined, {{ numeric: true, sensitivity: 'base' }});
                return currentSort.direction === 'asc' ? comparison : -comparison;
            }});
            
            // Reorder rows in DOM (including detail rows)
            rows.forEach((row, idx) => {{
                const detailRow = document.getElementById(row.id + '-detail');
                tbody.appendChild(row);
                if (detailRow) tbody.appendChild(detailRow);
            }});
            
            // Update allRows array
            allRows = rows;
        }}
        
        function toggleRow(rowId) {{
            const row = document.getElementById(rowId);
            const detailRow = document.getElementById(rowId + '-detail');
            if (row && detailRow) {{
                const isExpanded = row.dataset.expanded === 'true';
                const nextExpanded = !isExpanded;
                row.dataset.expanded = nextExpanded ? 'true' : 'false';
                detailRow.style.display = nextExpanded ? '' : 'none';
            }}
        }}
        
        function updateStats() {{
            const remoteCount = Array.from(allRows).filter(row => {{
                const cells = row.querySelectorAll('td');
                return cells[3].textContent.trim() === 'Yes';
            }}).length;
            document.getElementById('remote-count').textContent = remoteCount || '—';
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report generated: {output_path}")
    print(f"   Open in your browser to view {len(matches)} matches")


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
        "--no-extract",
        action="store_true",
        help="Skip LLM-based extraction of structured data (faster, but no extracted fields). Only used with --search-eng-management.",
    )
    parser.add_argument(
        "--input",
        default="posts.csv",
        help="Input file (CSV for --fetch-comments mode, JSON for --search-eng-management, --extract-from-matches, and --generate-html modes, default: posts.csv).",
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
        help="Path to write output (default: who_is_hiring_posts.csv for posts, comments.json for comments, matches.json for search, matches_with_extraction.json for extraction, report.html for HTML).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate_html:
        # Mode 5: Generate HTML report
        if not args.output.endswith(".html"):
            # Default to report.html if not specified
            if args.output == "who_is_hiring_posts.csv":
                args.output = "report.html"

        generate_html_report(args.input, args.output)
    elif args.extract_from_matches:
        # Mode 4: Extract structured data from existing matches
        if not args.output.endswith(".json"):
            # Default to matches_with_extraction.json if not specified
            if args.output == "who_is_hiring_posts.csv":
                args.output = "matches_with_extraction.json"

        # Safety check: don't overwrite comments.json
        if args.output == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches_with_extraction.json instead."
            )
            args.output = "matches_with_extraction.json"

        extract_from_matches(args.input, args.output)
    elif args.search_eng_management:
        # Mode 3: Search for engineering management roles
        if not args.output.endswith(".json"):
            # Default to matches.json if not specified
            if args.output == "who_is_hiring_posts.csv":
                args.output = "matches.json"

        # Safety check: don't overwrite comments.json
        if args.output == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches.json instead."
            )
            args.output = "matches.json"

        matches = search_engineering_management_roles(
            args.input, extract_with_llm=not args.no_extract
        )
        write_json(matches, args.output)
        print(f"\nWrote {len(matches)} matches to {args.output}")
    elif args.fetch_comments:
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
