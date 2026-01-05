#!/usr/bin/env python3
"""Extract structured job data using Instructor.

This module provides the Instructor-based extraction backend.
It mirrors the interface of extract_jobs_baml.py for interoperability.
"""

from __future__ import annotations

import html
import os
import re

import instructor
from openai import OpenAI
from pydantic import ValidationError

from utils import infer_provider


# Cached client to avoid recreating on each call
_cached_client: instructor.Instructor | None = None
_cached_model: str | None = None


def clean_html_content(content: str) -> str:
    """Clean HTML tags and decode entities for LLM processing."""
    # Decode HTML entities
    decoded = html.unescape(content)
    # Remove HTML tags but preserve content
    cleaned = re.sub(r"<[^>]+>", " ", decoded)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def create_instructor_client(model: str) -> instructor.Instructor:
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
        raise RuntimeError(
            f"Unsupported provider: {provider}. Use 'openai' or 'gemini'."
        )


def _get_client(model: str) -> instructor.Instructor:
    """Get or create a cached instructor client for the model."""
    global _cached_client, _cached_model

    # Invalidate cache if model changed (different provider may be needed)
    if _cached_client is None or _cached_model != model:
        _cached_client = create_instructor_client(model)
        _cached_model = model

    return _cached_client


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


def extract_from_comment_instructor(
    comment: dict,
    model: str,
):
    """Extract structured job data from a comment using Instructor.

    Args:
        comment: Comment dict with 'content' field
        model: Model to use (e.g., 'gpt-4o-mini')

    Returns:
        Tuple of (extraction_result, error, total_tokens).
    """
    from extract_jobs import CommentExtraction, ExtractionError

    content = comment.get("content", "")
    if not content or not content.strip():
        return (
            None,
            ExtractionError(
                error_type="empty_content",
                error_message="Comment has no content",
                retryable=False,
            ),
            None,
        )

    cleaned_content = clean_html_content(content)

    # Truncate if too long (stay within token limits)
    if len(cleaned_content) > 6000:
        cleaned_content = cleaned_content[:6000] + "..."

    try:
        client = _get_client(model)
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
        return (
            None,
            ExtractionError(
                error_type="validation_error",
                error_message=str(e),
                retryable=True,
            ),
            None,
        )
    except Exception as e:
        return (
            None,
            ExtractionError(
                error_type="api_error",
                error_message=str(e),
                retryable=True,
            ),
            None,
        )
