"""Search, extract, and report on job postings from 'Who is hiring?' comments.

Use sync_comments.py to download comments first, then use this script to:
- Search for roles matching a profile
- Extract structured data using LLM
- Generate HTML reports

Usage:
    python who_is_hiring.py --search --profile profiles/engineering_management.yaml
    python who_is_hiring.py --extract-from-matches --input out/engineering_management/matches.json
    python who_is_hiring.py --generate-html --input out/engineering_management/matches_with_extraction.json
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import webbrowser
from html.parser import HTMLParser
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

from sync_comments import load_cache

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"
DEFAULT_PROFILE_PATH = BASE_DIR / "profiles" / "engineering_management.yaml"
DEFAULT_COMMENTS_PATH = OUT_DIR / "comments.json"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


class AnchorPreservingSanitizer(HTMLParser):
    """Escape HTML while allowing safe <a> tags.

    HN comments sometimes include anchor tags (e.g., markdown links rendered by HN).
    We want those clickable in the report without allowing arbitrary HTML. This parser
    escapes everything except http(s)/mailto anchors, which are re-emitted with
    target/rel safety attributes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    @staticmethod
    def _is_safe_href(href: str) -> bool:
        lowered = href.lower()
        return (
            lowered.startswith("http://")
            or lowered.startswith("https://")
            or lowered.startswith("mailto:")
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


def profile_slug(profile_path: str | None) -> str:
    """Create a safe slug from a profile path (basename without extension)."""
    if not profile_path:
        return "engineering_management"
    slug = Path(profile_path).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)
    return slug or "profile"


def write_json(data: list[dict], output_path: str) -> None:
    """Write data to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)


def extract_job_info_with_llm(
    content: str, client: OpenAI | None = None, matched_text: str | None = None
) -> dict:
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


def build_match_key(
    comment_id: str | None,
    matched_text: str | None,
) -> str | None:
    """Return the dedupe key for a match: comment ID only."""

    if comment_id is None:
        return None
    cid = str(comment_id).strip()
    return cid or None


def load_existing_extractions(path: Path) -> dict[str, dict]:
    """Load prior extractions keyed by comment+match, skipping errored entries."""

    if not path.exists():
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            existing_matches = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read existing extractions from {path}: {exc}")
        return {}

    keyed: dict[str, dict] = {}
    if not isinstance(existing_matches, list):
        return keyed

    for entry in existing_matches:
        key = build_match_key(entry.get("comment_id"), entry.get("matched_text"))
        if not key:
            continue
        extracted = entry.get("extracted") or {}
        if extracted and not extracted.get("extraction_error"):
            keyed[key] = extracted
    return keyed


def compile_patterns_from_profile(profile_path: Path) -> list[tuple[re.Pattern, str]]:
    """Compile regex patterns for role search from a YAML profile."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with open(profile_path, encoding="utf-8") as f:
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
    profile_path: str | None = None,
    max_matches: int | None = None,
    existing_extracted_path: str | None = None,
) -> list[dict]:
    """Search comments for engineering management role postings.

    Args:
        comments_path: Path to JSON file containing comments
        extract_with_llm: Whether to extract structured data using LLM (default: True)
        profile_path: Path to YAML profile defining regex patterns (default: engineering management profile)
        existing_extracted_path: Path to a matches_with_extraction.json used to reuse prior extractions

    Returns:
        List of match dictionaries with comment info and matched text
    """
    # Load comments
    print(f"Loading comments from {comments_path}...")
    comments_cache = load_cache(Path(comments_path))
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

    existing_extractions: dict[str, dict] = {}
    if existing_extracted_path:
        existing_extractions = load_existing_extractions(Path(existing_extracted_path))
        if existing_extractions:
            print(
                f"Reusing {len(existing_extractions)} existing extractions from {existing_extracted_path}"
            )

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
                    key = build_match_key(
                        match_dict.get("comment_id"), match_dict.get("matched_text")
                    )
                    reused = False
                    if key and key in existing_extractions:
                        match_dict["extracted"] = existing_extractions[key]
                        reused = True
                    if not reused:
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


def extract_from_matches(
    input_path: str, output_path: str, reextract: bool = False
) -> None:
    """Read matches from JSON file and add extracted data to each match.

    Args:
        input_path: Path to input JSON file with matches
        output_path: Path to write output JSON file with extracted data
        reextract: Force rerun extraction even if match already has extracted data
    """
    print(f"Loading matches from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
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

    existing_extractions: dict[str, dict] = {}
    output_path_obj = Path(output_path)
    if output_path_obj.exists() and not reextract:
        existing_extractions = load_existing_extractions(output_path_obj)
        if existing_extractions:
            print(
                f"Reusing {len(existing_extractions)} existing extractions from {output_path_obj}"
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
            key = build_match_key(match.get("comment_id"), match.get("matched_text"))
            if key and key in existing_extractions:
                match["extracted"] = existing_extractions[key]
                print(
                    f"Match {idx}/{len(matches)} already extracted in {output_path_obj}; skipping (use --reextract to force)."
                )
                continue
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
                else "No"
                if extracted.get("is_remote") is False
                else "N/A"
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
    with open(input_path, encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")
    print("Generating HTML report...")

    # Format date for display
    def format_date(date_str: str | None) -> str:
        if not date_str:
            return ""
        try:
            dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt_obj.strftime("%Y-%m-%d")
        except Exception:
            return date_str

    rows: list[dict[str, str]] = []
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
                    else "remote-no"
                    if is_remote_val is False
                    else ""
                ),
                "employment": (extracted.get("employment_type") or "").strip(),
                "cash_comp": (extracted.get("cash_compensation") or "").strip(),
                "equity": (
                    "Yes"
                    if extracted.get("equity_compensation")
                    else "No"
                    if extracted.get("equity_compensation") is False
                    else "—"
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
        description="Search, extract, and report on job postings from 'Who is hiring?' comments. Use sync_comments.py to download comments first."
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Mode: search comments for roles using regex patterns defined in a profile.",
    )
    parser.add_argument(
        "--profile",
        help="Path to YAML profile defining role search patterns (default: profiles/engineering_management.yaml).",
    )
    parser.add_argument(
        "--extract-from-matches",
        action="store_true",
        help="Mode: extract structured data from existing matches JSON file using LLM.",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        help="Mode: generate a self-contained HTML report from matches with extraction data.",
    )
    parser.add_argument(
        "--no-open-report",
        action="store_true",
        help="Skip automatically opening the HTML report after generation.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip LLM-based extraction of structured data (faster, but no extracted fields). Only used with --search.",
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
        help="Input file path (defaults: out/comments.json for search, out/<profile>/matches.json for extract, out/<profile>/matches_with_extraction.json for HTML).",
    )
    parser.add_argument(
        "--output",
        help="Output file path (defaults: out/<profile>/matches.json for search, out/<profile>/matches_with_extraction.json for extract, out/<profile>/report.html for HTML).",
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

    slug = profile_slug(args.profile)
    default_matches_path = OUT_DIR / slug / "matches.json"
    default_matches_with_extraction_path = (
        OUT_DIR / slug / "matches_with_extraction.json"
    )
    default_report_path = OUT_DIR / slug / "report.html"

    if args.generate_html:
        # Generate HTML report
        input_path = args.input or str(default_matches_with_extraction_path)
        output_path = args.output or str(default_report_path)

        generate_html_report(
            input_path, output_path, open_browser=not args.no_open_report
        )
    elif args.extract_from_matches:
        # Extract structured data from existing matches
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
        # Search comments for roles using a profile
        input_path = args.input or str(DEFAULT_COMMENTS_PATH)
        output_path = args.output or str(default_matches_path)
        profile_arg = args.profile
        output_path_obj = Path(output_path)

        if profile_arg:
            profile_path_obj = Path(profile_arg)
            if not profile_path_obj.exists():
                available = sorted(
                    p.name for p in (BASE_DIR / "profiles").glob("*.yaml")
                )
                available_msg = (
                    "Available profiles: " + ", ".join(available)
                    if available
                    else "No profiles found in profiles/"
                )
                print(f"Error: Profile file not found: {profile_arg}. {available_msg}")
                sys.exit(1)

        # Safety check: don't overwrite comments.json
        if output_path_obj.name == "comments.json":
            print(
                "Error: Cannot write to comments.json (protected file). Using matches.json instead."
            )
            output_path_obj = default_matches_path
            output_path = str(default_matches_path)

        existing_extracted_path = (
            output_path_obj.with_name("matches_with_extraction.json")
            if args.output
            else default_matches_with_extraction_path
        )

        matches = search_engineering_management_roles(
            input_path,
            extract_with_llm=not args.no_extract,
            profile_path=profile_arg,
            max_matches=args.max_matches,
            existing_extracted_path=str(existing_extracted_path),
        )
        write_json(matches, output_path)
        print(f"\nWrote {len(matches)} matches to {output_path}")
    else:
        # No mode specified - show help
        print(
            "Error: Please specify a mode: --search, --extract-from-matches, or --generate-html"
        )
        print("Use sync_comments.py to download comments first.")
        print("\nRun with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
