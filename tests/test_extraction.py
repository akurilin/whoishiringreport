"""Eval suite for job extraction using real HN comments as golden standards.

These tests validate that extraction:
1. Correctly identifies job postings
2. Handles one-to-many (multiple roles per comment)
3. Extracts expected fields accurately

Test cases are defined in fixtures/eval_cases.json for easy human review.

Run with:
    pytest tests/test_extraction.py -v                       # Default (gpt-4o-mini)
    pytest tests/test_extraction.py -v --models gemini-2.0-flash-lite
    pytest tests/test_extraction.py -v --models gpt-4o-mini,gemini-2.0-flash-lite,gemini-2.5-flash-lite
"""

import json
import time
from pathlib import Path

from conftest import (
    ExtractionTiming,
    get_test_models,
    get_timing_report,
    record_test_result,
)

# --- FIXTURES ---


def load_eval_cases() -> list[dict]:
    """Load all test cases from JSON fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "eval_cases.json"
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


EVAL_CASES = load_eval_cases()

# Cache for instructor clients (one per model)
_client_cache: dict[str, object] = {}


def get_client(model: str):
    """Get or create an instructor client for a model."""
    if model not in _client_cache:
        from extract_jobs import create_instructor_client

        _client_cache[model] = create_instructor_client(model)
    return _client_cache[model]


# --- ASSERTION HELPERS ---


def check_field_assertion(
    actual_value, field_name: str, expected_value, role_context: str = ""
):
    """Check a single field assertion with operator support.

    Operators (in field_name suffix):
    - _contains: any item in expected list is substring of actual
    - _gte: actual >= expected
    - _lte: actual <= expected
    - _in: actual is in expected list
    - (none): exact match, or null check
    """
    ctx = f" for {role_context}" if role_context else ""

    # Handle _contains operator (substring matching)
    if field_name.endswith("_contains") or field_name.endswith("_contain"):
        base_field = field_name.rsplit("_", 1)[0]
        if actual_value is None:
            raise AssertionError(
                f"{base_field} is None, expected to contain {expected_value}{ctx}"
            )

        # actual_value could be a list (e.g., locations, remote_regions) or string
        if isinstance(actual_value, list):
            actual_str = " ".join(str(v) for v in actual_value)
        else:
            actual_str = str(actual_value)

        # expected_value is a list of possible substrings (any match passes)
        if not isinstance(expected_value, list):
            expected_value = [expected_value]

        if not any(substr in actual_str for substr in expected_value):
            raise AssertionError(
                f"{base_field} '{actual_str}' does not contain any of {expected_value}{ctx}"
            )
        return

    # Handle _gte operator (greater than or equal)
    if field_name.endswith("_gte"):
        base_field = field_name[:-4]
        if actual_value is None:
            raise AssertionError(
                f"{base_field} is None, expected >= {expected_value}{ctx}"
            )
        if actual_value < expected_value:
            raise AssertionError(f"{base_field} {actual_value} < {expected_value}{ctx}")
        return

    # Handle _lte operator (less than or equal)
    if field_name.endswith("_lte"):
        base_field = field_name[:-4]
        if actual_value is None:
            raise AssertionError(
                f"{base_field} is None, expected <= {expected_value}{ctx}"
            )
        if actual_value > expected_value:
            raise AssertionError(f"{base_field} {actual_value} > {expected_value}{ctx}")
        return

    # Handle _in operator (actual is in expected list)
    if field_name.endswith("_in"):
        base_field = field_name[:-3]
        if actual_value not in expected_value:
            raise AssertionError(
                f"{base_field} {actual_value} not in {expected_value}{ctx}"
            )
        return

    # Exact match (including null checks)
    if actual_value != expected_value:
        raise AssertionError(
            f"{field_name}: expected {expected_value}, got {actual_value}{ctx}"
        )


def get_role_field(role, field_name: str):
    """Get a field value from a role, handling operator suffixes."""
    # Strip operator suffixes to get base field name
    base_field = field_name
    for suffix in ("_contains", "_contain", "_gte", "_lte", "_in"):
        if base_field.endswith(suffix):
            base_field = base_field[: -len(suffix)]
            break

    # Handle nested access for enums
    value = getattr(role, base_field, None)

    # Convert enum to string value for comparison
    if hasattr(value, "value"):
        value = value.value

    return value


def find_matching_role(roles, match_by: dict):
    """Find a role that matches the match_by criteria (order-agnostic).

    Returns the matched role or None.
    """
    for role in roles:
        matches = True
        for field_name, expected in match_by.items():
            actual = get_role_field(role, field_name)
            try:
                check_field_assertion(actual, field_name, expected)
            except AssertionError:
                matches = False
                break
        if matches:
            return role
    return None


def assert_extraction_matches(result, error, expected: dict, case_name: str):
    """Assert that extraction result matches expected schema.

    Args:
        result: CommentExtraction or None
        error: ExtractionError or None
        expected: Expected schema dict from JSON
        case_name: Test case name for error messages
    """
    # Handle error cases
    if expected.get("error"):
        assert error is not None, f"[{case_name}] Expected an error but got none"
        if "error_type_contains" in expected:
            error_text = f"{error.error_type} {error.error_message}".lower()
            assert expected["error_type_contains"].lower() in error_text, (
                f"[{case_name}] Error should contain '{expected['error_type_contains']}', "
                f"got: {error.error_type}: {error.error_message}"
            )
        return

    # Non-error cases should have no error
    assert error is None, f"[{case_name}] Unexpected error: {error}"
    assert result is not None, f"[{case_name}] Result is None"

    # Check is_job_posting
    if "is_job_posting" in expected:
        assert result.is_job_posting == expected["is_job_posting"], (
            f"[{case_name}] is_job_posting: expected {expected['is_job_posting']}, "
            f"got {result.is_job_posting}"
        )

    # Check role count
    if "role_count" in expected:
        assert len(result.roles) == expected["role_count"], (
            f"[{case_name}] Expected {expected['role_count']} roles, got {len(result.roles)}"
        )

    if "role_count_min" in expected:
        assert len(result.roles) >= expected["role_count_min"], (
            f"[{case_name}] Expected at least {expected['role_count_min']} roles, "
            f"got {len(result.roles)}"
        )

    # Check all_roles constraints (applies to every role)
    if "all_roles" in expected:
        for i, role in enumerate(result.roles):
            role_ctx = f"role[{i}] ({role.role_title or 'untitled'})"
            for field_name, expected_value in expected["all_roles"].items():
                actual = get_role_field(role, field_name)
                check_field_assertion(actual, field_name, expected_value, role_ctx)

    # Check individual role expectations (order-agnostic matching)
    if "roles" in expected:
        for role_spec in expected["roles"]:
            match_by = role_spec.get("match_by", {})
            expect = role_spec.get("expect", {})

            # Find matching role
            matched_role = find_matching_role(result.roles, match_by)
            match_desc = str(match_by)

            assert matched_role is not None, (
                f"[{case_name}] No role matched {match_desc}. "
                f"Available roles: {[r.role_title for r in result.roles]}"
            )

            # Check expectations on matched role
            role_ctx = f"role matching {match_desc}"
            for field_name, expected_value in expect.items():
                actual = get_role_field(matched_role, field_name)
                check_field_assertion(actual, field_name, expected_value, role_ctx)


# --- TIMED EXTRACTION ---


def timed_extract(client, comment, model, case_name: str):
    """Wrapper that times extraction and records to timing report."""
    from extract_jobs import extract_from_comment

    print(f"\n  [{case_name}] Extracting with {model}...")
    start = time.time()
    result, error, total_tokens = extract_from_comment(client, comment, model=model)
    elapsed = time.time() - start

    # Record timing
    role_count = len(result.roles) if result else 0
    timing = ExtractionTiming(
        case_name=case_name,
        model=model,
        elapsed_seconds=elapsed,
        success=error is None,
        role_count=role_count,
        error_type=error.error_type if error else None,
        total_tokens=total_tokens,
    )
    get_timing_report(model).extractions.append(timing)

    tokens_str = f", {total_tokens} tokens" if total_tokens else ""
    if error:
        print(f"  [{case_name}] FAILED in {elapsed:.2f}s - {error.error_type}")
    else:
        print(f"  [{case_name}] OK in {elapsed:.2f}s - {role_count} roles{tokens_str}")

    return result, error


# --- DYNAMIC TEST GENERATION ---


def pytest_generate_tests(metafunc):
    """Generate test parameters based on CLI options."""
    if "case" in metafunc.fixturenames and "model" in metafunc.fixturenames:
        models = get_test_models(metafunc.config)

        # Generate tests for all models × all cases
        params = [(case, model) for model in models for case in EVAL_CASES]

        if len(models) > 1:
            ids = [f"{model}::{case['name']}" for model in models for case in EVAL_CASES]
        else:
            ids = [case["name"] for case in EVAL_CASES]

        metafunc.parametrize("case,model", params, ids=ids)


# --- TESTS ---


def test_extraction(case, model):
    """Run extraction test case from JSON fixture."""
    client = get_client(model)
    result, error = timed_extract(client, case["comment"], model, case["name"])

    try:
        assert_extraction_matches(result, error, case["expected"], case["name"])
        record_test_result(model, passed=True)
    except AssertionError:
        record_test_result(model, passed=False)
        raise
