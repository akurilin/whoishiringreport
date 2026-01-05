#!/usr/bin/env python3
"""Extract structured job data using BAML.

This module mirrors extract_jobs.py but uses BAML for LLM extraction.
It provides the same interface for interoperability with the eval suite.
"""

from __future__ import annotations

import html
import re

from baml_client.sync_client import b
from baml_client.types import CommentExtraction as BamlCommentExtraction
from baml_client.types import CompanyStage as BamlCompanyStage
from baml_client.types import EmploymentType as BamlEmploymentType
from baml_client.types import ExtractedRole as BamlExtractedRole

# Map model names to BAML client names
MODEL_TO_CLIENT = {
    "gpt-4o-mini": "GPT4oMini",
    "gpt-4o": "GPT4o",
    "gemini-2.0-flash-lite": "Gemini2FlashLite",
    "gemini-2.5-flash-lite": "Gemini25FlashLite",
    "gemini-2.0-flash": "Gemini2Flash",
}

# Map BAML enum values to instructor-compatible string values
EMPLOYMENT_TYPE_MAP = {
    BamlEmploymentType.FullTime: "Full-time",
    BamlEmploymentType.PartTime: "Part-time",
    BamlEmploymentType.Contract: "Contract",
    BamlEmploymentType.Internship: "Internship",
    BamlEmploymentType.Fractional: "Fractional",
}

COMPANY_STAGE_MAP = {
    BamlCompanyStage.PreSeed: "Pre-seed",
    BamlCompanyStage.Seed: "Seed",
    BamlCompanyStage.SeriesA: "Series A",
    BamlCompanyStage.SeriesB: "Series B",
    BamlCompanyStage.SeriesC: "Series C",
    BamlCompanyStage.SeriesDPlus: "Series D+",
    BamlCompanyStage.Public: "Public",
    BamlCompanyStage.Bootstrapped: "Bootstrapped",
}


def get_baml_client_name(model: str) -> str:
    """Map model name to BAML client name.

    Args:
        model: Model name (e.g., 'gpt-4o-mini')

    Returns:
        BAML client name (e.g., 'GPT4oMini')

    Raises:
        ValueError: If model is not supported
    """
    if model not in MODEL_TO_CLIENT:
        raise ValueError(
            f"Model '{model}' not configured in BAML. "
            f"Supported models: {list(MODEL_TO_CLIENT.keys())}"
        )
    return MODEL_TO_CLIENT[model]


def clean_html_content(content: str) -> str:
    """Clean HTML tags and decode entities for LLM processing."""
    # Decode HTML entities
    decoded = html.unescape(content)
    # Remove HTML tags but preserve content
    cleaned = re.sub(r"<[^>]+>", " ", decoded)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def convert_baml_role(baml_role: BamlExtractedRole) -> dict:
    """Convert BAML ExtractedRole to dict with instructor-compatible values."""
    return {
        "role_title": baml_role.role_title,
        "locations": baml_role.locations,
        "is_remote": baml_role.is_remote,
        "remote_regions": baml_role.remote_regions,
        "employment_type": (
            EMPLOYMENT_TYPE_MAP.get(baml_role.employment_type)
            if baml_role.employment_type
            else None
        ),
        "salary_min": baml_role.salary_min,
        "salary_max": baml_role.salary_max,
        "salary_currency": baml_role.salary_currency,
        "salary_raw": baml_role.salary_raw,
        "equity": baml_role.equity,
        "application_method": baml_role.application_method,
        "company_name": baml_role.company_name,
        "company_stage": (
            COMPANY_STAGE_MAP.get(baml_role.company_stage)
            if baml_role.company_stage
            else None
        ),
        "company_url": baml_role.company_url,
    }


def convert_baml_extraction(baml_result: BamlCommentExtraction):
    """Convert BAML result to instructor-compatible CommentExtraction.

    This allows the BAML output to work with the existing test assertions
    and output format.

    Returns:
        CommentExtraction from extract_jobs module
    """
    from extract_jobs import CommentExtraction, ExtractedRole

    # Convert each role
    roles = []
    for baml_role in baml_result.roles:
        role_dict = convert_baml_role(baml_role)
        # Create ExtractedRole from dict - validators will handle coercion
        roles.append(ExtractedRole(**role_dict))

    return CommentExtraction(
        roles=roles,
        is_job_posting=baml_result.is_job_posting,
        extraction_confidence=baml_result.extraction_confidence,
    )


def extract_from_comment_baml(
    comment: dict,
    model: str,
):
    """Extract structured job data from a comment using BAML.

    Args:
        comment: Comment dict with 'content' field
        model: Model to use (e.g., 'gpt-4o-mini')

    Returns:
        Tuple of (extraction_result, error, total_tokens).
        BAML doesn't expose token counts directly, so tokens is always None.
    """
    from extract_jobs import ExtractionError

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
        # Get BAML client name for this model
        client_name = get_baml_client_name(model)

        # Call BAML extraction with runtime client override
        baml_result = b.ExtractJobData(
            comment_content=cleaned_content,
            baml_options={"client": client_name},
        )

        # Convert to instructor-compatible format
        result = convert_baml_extraction(baml_result)

        return result, None, None  # BAML doesn't expose token counts

    except ValueError as e:
        # Model not configured
        return (
            None,
            ExtractionError(
                error_type="configuration_error",
                error_message=str(e),
                retryable=False,
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
