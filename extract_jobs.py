#!/usr/bin/env python3
"""Extract structured job data from HN 'Who is hiring?' comments.

Uses Pydantic + Instructor for type-safe LLM extraction with one-to-many support.
Outputs flat JSON with one row per job role (denormalized).

Usage:
    python extract_jobs.py                    # Extract all unprocessed comments
    python extract_jobs.py --limit 100        # Extract first 100 unprocessed
    python extract_jobs.py --refresh          # Re-extract all comments
    python extract_jobs.py --output X         # Custom output path
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Literal

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from sync_comments import load_cache

# Load environment variables
load_dotenv()

# --- CONSTANTS ---
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"
DEFAULT_COMMENTS_PATH = OUT_DIR / "comments.json"
DEFAULT_OUTPUT_PATH = OUT_DIR / "extracted_jobs.json"
DEFAULT_PROVIDER = "openai"
SCHEMA_VERSION = 1
BATCH_SIZE = 50  # Save progress every N extractions
RATE_LIMIT_DELAY = 0.15  # seconds between API calls

# Default model
DEFAULT_MODEL = "gpt-4o-mini"


def infer_provider(model: str) -> str:
    """Infer the provider from the model name.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'gemini-2.0-flash-lite')

    Returns:
        Provider name ('openai' or 'gemini')
    """
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    elif model.startswith("gemini-"):
        return "gemini"
    else:
        # Default to OpenAI for unknown models
        return "openai"


# --- ENUMS ---


class EmploymentType(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    CONTRACT = "Contract"
    INTERNSHIP = "Internship"
    FRACTIONAL = "Fractional"

    @classmethod
    def _missing_(cls, value):
        """Handle string values that don't exactly match enum values."""
        if isinstance(value, str):
            # Try case-insensitive match
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class CompanyStage(str, Enum):
    PRE_SEED = "Pre-seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    SERIES_D_PLUS = "Series D+"
    PUBLIC = "Public"
    BOOTSTRAPPED = "Bootstrapped"

    @classmethod
    def _missing_(cls, value):
        """Handle string values that don't exactly match enum values."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


# --- PYDANTIC MODELS ---


class ExtractedRole(BaseModel):
    """A single job role extracted from a comment (flat, denormalized with company info)."""

    # Role details
    role_title: str | None = Field(
        default=None,
        description="The job title, e.g., 'Senior Backend Engineer', 'Engineering Manager'",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Physical locations for this role, e.g., ['San Francisco', 'NYC', 'Remote']",
    )
    is_remote: bool | None = Field(
        default=None,
        description="Whether fully remote work is available",
    )
    remote_regions: list[str] = Field(
        default_factory=list,
        description="Geographic restrictions for remote, e.g., ['US only', 'North America', 'EU']",
    )
    employment_type: EmploymentType | None = Field(
        default=None,
        description="Type of employment: Full-time, Part-time, Contract, Internship, Fractional",
    )
    salary_min: int | None = Field(
        default=None,
        description="Minimum salary as integer (e.g., 150000 for $150k)",
    )
    salary_max: int | None = Field(
        default=None,
        description="Maximum salary as integer (e.g., 250000 for $250k)",
    )
    salary_currency: str | None = Field(
        default=None,
        description="Currency code, e.g., 'USD', 'EUR', 'GBP'. Default to USD for $ amounts.",
    )
    salary_raw: str | None = Field(
        default=None,
        description="Original salary text as written, e.g., '$150k-$250k'",
    )
    equity: str | None = Field(
        default=None,
        description="Equity amount as percentage, e.g., '0.5%', '1-2%', '0.1-0.5%'. Null if not specified or just mentioned without amount.",
    )
    application_method: str | None = Field(
        default=None,
        description="How to apply: email address, URL, or instructions",
    )

    # Company details (denormalized per role)
    company_name: str | None = Field(
        default=None,
        description="Name of the hiring company",
    )
    company_stage: CompanyStage | None = Field(
        default=None,
        description="Funding stage: Pre-seed, Seed, Series A/B/C, Series D+, Public, Bootstrapped",
    )
    company_url: str | None = Field(
        default=None,
        description="Company website URL if mentioned",
    )

    @field_validator("locations", "remote_regions", mode="before")
    @classmethod
    def convert_none_to_list(cls, v):
        """Convert None to empty list for list fields (handles smaller models returning null)."""
        return v if v is not None else []

    @field_validator("employment_type", mode="before")
    @classmethod
    def coerce_employment_type(cls, v):
        """Coerce string to EmploymentType enum (handles Gemini returning raw strings)."""
        if v is None or isinstance(v, EmploymentType):
            return v
        if isinstance(v, str):
            # Try exact match first
            try:
                return EmploymentType(v)
            except ValueError:
                pass
            # Try case-insensitive match
            for member in EmploymentType:
                if member.value.lower() == v.lower():
                    return member
        return None

    @field_validator("company_stage", mode="before")
    @classmethod
    def coerce_company_stage(cls, v):
        """Coerce string to CompanyStage enum (handles Gemini returning raw strings)."""
        if v is None or isinstance(v, CompanyStage):
            return v
        if isinstance(v, str):
            try:
                return CompanyStage(v)
            except ValueError:
                pass
            for member in CompanyStage:
                if member.value.lower() == v.lower():
                    return member
        return None


class CommentExtraction(BaseModel):
    """LLM extraction result for a single comment (may contain multiple roles)."""

    roles: list[ExtractedRole] = Field(
        default_factory=list,
        description="List of job roles extracted from this comment. One comment can have multiple roles.",
    )
    is_job_posting: bool = Field(
        default=False,
        description="Whether this comment is actually a job posting (vs noise/replies/off-topic)",
    )
    extraction_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence in the extraction quality",
    )


class ExtractionError(BaseModel):
    """Represents a failed extraction attempt."""

    error_type: str
    error_message: str
    retryable: bool = True


# --- INSTRUCTOR CLIENT ---


def create_instructor_client(model: str = DEFAULT_MODEL) -> instructor.Instructor:
    """Create an Instructor-wrapped client for the given model.

    Provider is automatically inferred from the model name.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'gemini-2.0-flash-lite')

    Returns:
        Instructor client configured for the provider

    Raises:
        RuntimeError: If API key is missing or provider is unsupported
    """
    provider = infer_provider(model)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        return instructor.from_openai(OpenAI(api_key=api_key))

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")
        # Use the new google.genai SDK
        from google import genai
        from instructor import from_genai

        client = genai.Client(api_key=api_key)
        return from_genai(client, model=model)

    else:
        raise RuntimeError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")


# --- EXTRACTION LOGIC ---


def clean_html_content(content: str) -> str:
    """Clean HTML tags and decode entities for LLM processing."""
    # Decode HTML entities
    decoded = html.unescape(content)
    # Remove HTML tags but preserve content
    cleaned = re.sub(r"<[^>]+>", " ", decoded)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_extraction_prompt() -> str:
    """Build the system prompt for extraction."""
    return """You are a job posting data extractor for Hacker News "Who is hiring?" comments.

CRITICAL RULES:
1. One comment may contain MULTIPLE job roles. Extract ALL distinct roles mentioned.
2. If a role is not a job posting (e.g., a reply, question, or off-topic comment), set is_job_posting=false and return empty roles.
3. For multi-role comments, each role should have the SAME company info but may have different titles/levels.

EXAMPLES OF MULTIPLE ROLES:
- "We're hiring: Senior Backend Engineer, Staff Frontend Engineer, Engineering Manager" -> 3 roles
- "Looking for: Go developers (junior and senior levels)" -> 2 roles (Junior and Senior)
- Bullet-pointed or dash-listed positions -> one role per bullet/dash

EXTRACTION RULES:
- salary_min/salary_max: Parse from ranges. "$150k-$250k" -> 150000, 250000. "$170-225K" -> 170000, 225000.
- salary_currency: Default to "USD" for $ amounts unless explicitly specified otherwise.
- is_remote: True if "remote", "remote-first", "remote-friendly", "WFH", "work from home". False if "onsite only", "in-person required".
- remote_regions: Extract geographic restrictions like "US only", "North America", "EU timezone", "EMEA".
- employment_type: Normalize to enum values. "Full Time" -> "Full-time", "FT" -> "Full-time".
- company_stage: Look for "Series A/B/C", "Seed", "bootstrapped", "public company", funding amounts.
- application_method: Extract email addresses or URLs mentioned for applying.

ROLE TITLE RULES:
- Use the exact title mentioned when possible
- If multiple seniority levels for same role, create separate entries: "Senior SWE" and "Staff SWE"
- "SWE" = "Software Engineer", "MLE" = "Machine Learning Engineer"

OUTPUT QUALITY:
- If information is not mentioned, use null (not empty string)
- Set extraction_confidence based on how clear the posting is
- is_job_posting should be false for: comments asking questions, replies to other posts, meta-discussion"""


def get_total_tokens(completion) -> int | None:
    """Extract total token count from completion object.

    Handles differences between OpenAI and Gemini response formats.
    """
    if completion is None:
        return None

    # OpenAI format: completion.usage.total_tokens
    if hasattr(completion, "usage") and completion.usage is not None:
        return getattr(completion.usage, "total_tokens", None)

    # Gemini format: completion.usage_metadata.total_token_count
    if hasattr(completion, "usage_metadata") and completion.usage_metadata is not None:
        return getattr(completion.usage_metadata, "total_token_count", None)

    return None


def extract_from_comment(
    client: instructor.Instructor,
    comment: dict,
    model: str = DEFAULT_MODEL,
) -> tuple[CommentExtraction | None, ExtractionError | None, int | None]:
    """Extract structured job data from a single comment.

    Args:
        client: Instructor-wrapped OpenAI client
        comment: Comment dict with 'content' field
        model: Model to use

    Returns:
        Tuple of (extraction_result, error, total_tokens). Error or result will be None.
    """
    content = comment.get("content", "")
    if not content or not content.strip():
        return None, ExtractionError(
            error_type="empty_content",
            error_message="Comment has no content",
            retryable=False,
        ), None

    cleaned_content = clean_html_content(content)

    # Truncate if too long (stay within token limits)
    if len(cleaned_content) > 6000:
        cleaned_content = cleaned_content[:6000] + "..."

    try:
        extraction, completion = client.chat.completions.create_with_completion(
            model=model,
            response_model=CommentExtraction,
            messages=[
                {"role": "system", "content": build_extraction_prompt()},
                {
                    "role": "user",
                    "content": f"Extract job data from this HN comment:\n\n{cleaned_content}",
                },
            ],
            max_retries=2,
        )
        total_tokens = get_total_tokens(completion)
        return extraction, None, total_tokens

    except ValidationError as e:
        return None, ExtractionError(
            error_type="validation_error",
            error_message=str(e),
            retryable=True,
        ), None
    except Exception as e:
        return None, ExtractionError(
            error_type="api_error",
            error_message=str(e),
            retryable=True,
        ), None


# --- CACHING AND INCREMENTAL PROCESSING ---


def load_extraction_cache(path: Path) -> dict:
    """Load existing extraction results for incremental processing."""
    if not path.exists():
        return {
            "items": [],
            "metadata": {
                "last_extracted_at": None,
                "schema_version": SCHEMA_VERSION,
                "total_comments_processed": 0,
                "total_roles_extracted": 0,
                "extraction_errors": 0,
            },
        }

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_processed_comment_ids(cache: dict) -> set[str]:
    """Get set of already-extracted comment IDs from flat rows."""
    return {
        str(item.get("comment_id"))
        for item in cache.get("items", [])
        if item.get("comment_id")
    }


def save_extraction_cache(cache: dict, path: Path) -> None:
    """Save extraction results to disk."""
    cache["metadata"]["last_extracted_at"] = dt.datetime.now(dt.UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def flatten_extraction_to_rows(
    comment: dict,
    extraction: CommentExtraction,
    extracted_at: str,
) -> list[dict]:
    """Flatten a CommentExtraction into individual role rows (denormalized).

    Args:
        comment: Original comment dict
        extraction: The CommentExtraction result
        extracted_at: ISO timestamp of extraction

    Returns:
        List of flat dicts, one per role. If no roles, returns empty list.
    """
    if not extraction.roles:
        return []

    rows = []
    for role in extraction.roles:
        row = {
            # Comment metadata
            "comment_id": comment.get("id"),
            "post_url": comment.get("post_url", ""),
            "commenter": comment.get("commenter", ""),
            "comment_date": comment.get("date", ""),
            "raw_content": comment.get("content", ""),
            # Role data (from Pydantic model)
            **role.model_dump(),
            # Extraction metadata
            "is_job_posting": extraction.is_job_posting,
            "extraction_confidence": extraction.extraction_confidence,
            "extracted_at": extracted_at,
        }
        rows.append(row)

    return rows


# --- MAIN EXTRACTION LOOP ---


def extract_jobs(
    comments_path: Path,
    output_path: Path,
    limit: int | None = None,
    refresh: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Main extraction function.

    Args:
        comments_path: Path to comments.json
        output_path: Path to write extraction results
        limit: Maximum comments to process (None = all)
        refresh: If True, re-extract all comments
        model: Model to use (provider is inferred from model name)

    Returns:
        The updated extraction cache
    """
    # Load comments
    print(f"Loading comments from {comments_path}...")
    comments_cache = load_cache(comments_path)
    comments = comments_cache.get("items", [])
    print(f"Loaded {len(comments)} comments")

    # Load existing extractions
    extraction_cache = (
        {"items": [], "metadata": {"schema_version": SCHEMA_VERSION}}
        if refresh
        else load_extraction_cache(output_path)
    )
    extracted_ids = set() if refresh else get_processed_comment_ids(extraction_cache)

    print(f"Already extracted: {len(extracted_ids)} comments")

    # Filter to unprocessed comments
    pending_comments = [
        c for c in comments if c.get("id") and str(c["id"]) not in extracted_ids
    ]

    if limit:
        pending_comments = pending_comments[:limit]

    print(f"Processing {len(pending_comments)} comments...")

    if not pending_comments:
        print("No new comments to process.")
        return extraction_cache

    # Initialize client
    provider = infer_provider(model)
    client = create_instructor_client(model)
    print(f"Using model: {model} (provider: {provider})")

    # Process comments
    new_rows: list[dict] = []
    error_count = 0
    roles_count = 0
    processed_count = 0

    for idx, comment in enumerate(pending_comments, 1):
        comment_id = str(comment.get("id", ""))

        if idx % 10 == 0 or idx == 1:
            print(
                f"  Processing {idx}/{len(pending_comments)} (comment {comment_id})..."
            )

        extraction, error, _tokens = extract_from_comment(client, comment, model)
        extracted_at = dt.datetime.now(dt.UTC).isoformat()

        if error:
            print(f"  Error on comment {comment_id}: {error.error_message}")
            error_count += 1
            # Record error as a row with no role data
            new_rows.append(
                {
                    "comment_id": comment_id,
                    "post_url": comment.get("post_url", ""),
                    "commenter": comment.get("commenter", ""),
                    "comment_date": comment.get("date", ""),
                    "raw_content": comment.get("content", ""),
                    "extraction_error": error.model_dump(),
                    "extracted_at": extracted_at,
                }
            )
        else:
            # Flatten extraction to rows
            rows = flatten_extraction_to_rows(comment, extraction, extracted_at)
            roles_count += len(rows)
            new_rows.extend(rows)

            # Also record non-job-postings (no roles but processed)
            if not rows and extraction:
                new_rows.append(
                    {
                        "comment_id": comment_id,
                        "post_url": comment.get("post_url", ""),
                        "commenter": comment.get("commenter", ""),
                        "comment_date": comment.get("date", ""),
                        "is_job_posting": extraction.is_job_posting,
                        "extraction_confidence": extraction.extraction_confidence,
                        "extracted_at": extracted_at,
                    }
                )

        processed_count += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Periodic save
        if idx % BATCH_SIZE == 0:
            extraction_cache["items"].extend(new_rows)
            new_rows = []
            save_extraction_cache(extraction_cache, output_path)
            print(f"  Checkpoint saved ({idx} processed, {roles_count} roles)")

    # Final save
    extraction_cache["items"].extend(new_rows)
    extraction_cache["metadata"]["total_comments_processed"] = len(
        get_processed_comment_ids(extraction_cache)
    )
    extraction_cache["metadata"]["total_roles_extracted"] = sum(
        1 for item in extraction_cache["items"] if item.get("role_title")
    )
    extraction_cache["metadata"]["extraction_errors"] = sum(
        1 for item in extraction_cache["items"] if item.get("extraction_error")
    )

    save_extraction_cache(extraction_cache, output_path)

    print("\nExtraction complete!")
    print(f"  Comments processed: {processed_count}")
    print(f"  Roles extracted: {roles_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {output_path}")

    return extraction_cache


# --- CLI ---


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract structured job data from HN 'Who is hiring?' comments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_COMMENTS_PATH,
        help=f"Path to comments.json (default: {DEFAULT_COMMENTS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of comments to process",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-extract all comments (ignore existing extractions)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}). Provider is inferred from model name.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate API key upfront based on model
    provider = infer_provider(args.model)
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.", file=sys.stderr)
        print("Set it in your shell or .env file.", file=sys.stderr)
        sys.exit(1)
    elif provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment.", file=sys.stderr)
        print("Set it in your shell or .env file.", file=sys.stderr)
        sys.exit(1)

    extract_jobs(
        comments_path=args.input,
        output_path=args.output,
        limit=args.limit,
        refresh=args.refresh,
        model=args.model,
    )


if __name__ == "__main__":
    main()
