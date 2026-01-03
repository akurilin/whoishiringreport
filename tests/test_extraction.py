"""Eval suite for job extraction using real HN comments as golden standards.

These tests validate that extraction:
1. Correctly identifies job postings
2. Handles one-to-many (multiple roles per comment)
3. Extracts expected fields accurately

Run with: pytest tests/test_extraction.py -v
"""

import json
import os
import time
from pathlib import Path

import pytest

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping extraction eval tests",
)


@pytest.fixture(scope="module")
def eval_comments() -> dict:
    """Load the 3 golden test comments."""
    fixture_path = Path(__file__).parent / "fixtures" / "eval_comments.json"
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def instructor_client():
    """Create an Instructor-wrapped OpenAI client for testing."""
    from extract_jobs import create_instructor_client

    return create_instructor_client()


# Model to use for tests - change this to compare different models
TEST_MODEL = "gpt-4o-mini"


def timed_extract(client, comment, model):
    """Wrapper that times extraction and prints duration."""
    from extract_jobs import extract_from_comment

    test_name = comment.get("id", "unknown")
    print(f"\n  [{test_name}] Starting extraction with {model}...")
    start = time.time()
    result, error = extract_from_comment(client, comment, model=model)
    elapsed = time.time() - start

    if error:
        print(f"  [{test_name}] FAILED in {elapsed:.2f}s - {error.error_type}")
    else:
        role_count = len(result.roles) if result else 0
        print(
            f"  [{test_name}] Completed in {elapsed:.2f}s - {role_count} roles extracted"
        )

    return result, error, elapsed


class TestExtractionEval:
    """Golden standard eval suite for extraction quality."""

    def test_single_role_complete_data(self, eval_comments, instructor_client):
        """Case 1: Neon Health - single role with full data.

        Expected: 1 role with salary, location, remote, equity all present.
        """
        comment = eval_comments["44159535"]
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is None, f"Extraction failed: {error}"
        assert result is not None
        assert result.is_job_posting is True
        assert len(result.roles) == 1

        role = result.roles[0]
        assert role.company_name == "Neon Health"
        assert "Founding" in role.role_title or "Backend" in role.role_title
        assert role.is_remote is True
        assert any("North America" in r for r in role.remote_regions)
        assert role.salary_min == 170000
        assert role.salary_max == 225000
        assert role.salary_currency == "USD"
        # Equity mentioned but no specific amount -> should be null
        assert role.equity is None
        assert role.employment_type is not None
        # Check the enum value, not the string representation
        assert role.employment_type.value == "Full-time"

    def test_multi_role_extraction(self, eval_comments, instructor_client):
        """Case 2: SmarterDx - multiple roles from one comment.

        Expected: 5+ distinct roles, all sharing company info.
        Roles mentioned: Staff SWE, Senior SWE, ML Engineers, Engineering Manager,
        Analytics Engineers, Senior Security Engineer
        """
        comment = eval_comments["44159539"]
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is None, f"Extraction failed: {error}"
        assert result is not None
        assert result.is_job_posting is True
        assert len(result.roles) >= 5, (
            f"Expected at least 5 roles, got {len(result.roles)}"
        )

        # Check role diversity
        role_titles = [r.role_title.lower() for r in result.roles if r.role_title]
        assert any("staff" in t for t in role_titles), "Should have Staff role"
        assert any("senior" in t for t in role_titles), "Should have Senior role"
        assert any("ml" in t or "machine learning" in t for t in role_titles), (
            "Should have ML role"
        )
        assert any("manager" in t for t in role_titles), "Should have Manager role"

        # All roles should share company info
        for role in result.roles:
            assert role.company_name == "SmarterDx"
            assert role.is_remote is True
            assert any("US" in str(r) for r in role.remote_regions)
            # Salary should be consistent across roles (150-250k)
            if role.salary_min is not None:
                assert role.salary_min >= 150000
            if role.salary_max is not None:
                assert role.salary_max <= 260000  # Allow some variance

    def test_founding_roles_sparse_data(self, eval_comments, instructor_client):
        """Case 3: Galaxy - 3 founding roles, no salary.

        Expected: Exactly 3 roles (Frontend, Backend, ML), no salary info.
        """
        comment = eval_comments["44159626"]
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is None, f"Extraction failed: {error}"
        assert result is not None
        assert result.is_job_posting is True
        assert len(result.roles) == 3, (
            f"Expected exactly 3 roles, got {len(result.roles)}"
        )

        role_titles = [r.role_title for r in result.roles if r.role_title]
        assert any("Frontend" in t for t in role_titles), "Should have Frontend role"
        assert any("Backend" in t for t in role_titles), "Should have Backend role"
        assert any("Machine Learning" in t or "ML" in t for t in role_titles), (
            "Should have ML role"
        )

        # Salary not mentioned - should be null for all roles
        for role in result.roles:
            assert role.salary_min is None, (
                f"Salary should be null, got {role.salary_min}"
            )
            assert role.salary_max is None, (
                f"Salary should be null, got {role.salary_max}"
            )
            # Location should be NYC
            assert any("NYC" in loc or "NY" in loc for loc in role.locations), (
                f"Location should include NYC, got {role.locations}"
            )
            # Company name
            assert role.company_name == "Galaxy"
            # Not remote (onsite in NYC)
            assert role.is_remote is False or role.is_remote is None

    def test_single_role_principal(self, eval_comments, instructor_client):
        """Case 4: Snout - single principal-level role, remote US.

        Expected: 1 role with correct title, remote, full-time.
        """
        comment = eval_comments["44159532"]
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is None, f"Extraction failed: {error}"
        assert result is not None
        assert result.is_job_posting is True
        assert len(result.roles) == 1

        role = result.roles[0]
        assert role.company_name == "Snout"
        assert "Principal" in role.role_title or "Software" in role.role_title
        assert role.is_remote is True
        assert role.employment_type is not None
        assert role.employment_type.value == "Full-time"

    def test_single_role_remote_with_salary(self, eval_comments, instructor_client):
        """Case 5: Goody - single remote role with salary range.

        Expected: 1 role, fully remote, $170-230K salary.
        """
        comment = eval_comments["44159548"]
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is None, f"Extraction failed: {error}"
        assert result is not None
        assert result.is_job_posting is True
        assert len(result.roles) == 1

        role = result.roles[0]
        assert role.company_name == "Goody"
        assert "Senior" in role.role_title or "Software" in role.role_title
        assert role.is_remote is True
        assert role.salary_min == 170000
        assert role.salary_max == 230000
        assert role.salary_currency == "USD"
        # Equity mentioned but no specific amount -> should be null
        assert role.equity is None


class TestExtractionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self, instructor_client):
        """Should handle empty content gracefully."""
        comment = {"id": "empty_test", "content": ""}
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        assert error is not None
        assert (
            "empty" in error.error_type.lower()
            or "empty" in error.error_message.lower()
        )

    def test_non_job_posting(self, instructor_client):
        """Should identify non-job-posting comments."""
        comment = {
            "id": "non_job_test",
            "content": "Great thread! Looking forward to seeing the opportunities.",
        }
        result, error, elapsed = timed_extract(instructor_client, comment, TEST_MODEL)

        # Should succeed but mark as not a job posting
        if result is not None:
            assert result.is_job_posting is False
            assert len(result.roles) == 0
