"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest before running tests.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import infer_provider  # noqa: E402 - must be after path setup

# Load .env file before any tests run
# This ensures API keys are available for skip checks
load_dotenv()

# Default model for tests
DEFAULT_TEST_MODEL = "gpt-4o-mini"

# Stats file for historical analysis
STATS_FILE = Path(__file__).parent.parent / "out" / "eval_stats.jsonl"


def pytest_addoption(parser):
    """Add custom command-line options for extraction tests."""
    parser.addoption(
        "--models",
        action="store",
        default=DEFAULT_TEST_MODEL,
        help=f"Comma-separated list of models to test (default: {DEFAULT_TEST_MODEL})",
    )
    parser.addoption(
        "--extractors",
        action="store",
        default="instructor",
        help="Comma-separated list of extractors to test: instructor,baml (default: instructor)",
    )


def get_test_models(config) -> list[str]:
    """Get the list of models to test based on CLI options."""
    models_str = config.getoption("--models")
    return [m.strip() for m in models_str.split(",")]


def get_test_extractors(config) -> list[str]:
    """Get the list of extractors to test based on CLI options."""
    extractors_str = config.getoption("--extractors")
    return [e.strip() for e in extractors_str.split(",")]


def pytest_configure(config):
    """Configure pytest for eval suite."""
    # Validate that the correct API keys are set
    models = get_test_models(config)

    needs_openai = any(infer_provider(m) == "openai" for m in models)
    needs_gemini = any(infer_provider(m) == "gemini" for m in models)

    if needs_openai and not os.getenv("OPENAI_API_KEY"):
        raise pytest.UsageError(
            f"OPENAI_API_KEY not set (required for models: {[m for m in models if infer_provider(m) == 'openai']})"
        )
    if needs_gemini and not os.getenv("GEMINI_API_KEY"):
        raise pytest.UsageError(
            f"GEMINI_API_KEY not set (required for models: {[m for m in models if infer_provider(m) == 'gemini']})"
        )

    # Suppress traceback style (eval failures are expected, not bugs)
    config.option.tbstyle = "no"


def pytest_collection_finish(session):
    """Announce stats file location when running eval tests."""
    has_eval_tests = any("test_extraction" in item.nodeid for item in session.items)
    if has_eval_tests:
        print(f"\nStats will be saved to: {STATS_FILE}\n")


# --- TIMING INFRASTRUCTURE ---


@dataclass
class ExtractionTiming:
    """Tracks timing for a single extraction."""

    case_name: str
    model: str
    extractor: str  # "instructor" or "baml"
    elapsed_seconds: float
    success: bool
    role_count: int = 0
    error_type: str | None = None
    total_tokens: int | None = None


@dataclass
class TimingReport:
    """Aggregates timing data across all extractions."""

    key: str  # "extractor::model" combination
    extractions: list[ExtractionTiming] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0

    @property
    def total_time(self) -> float:
        return sum(e.elapsed_seconds for e in self.extractions)

    @property
    def avg_time(self) -> float:
        if not self.extractions:
            return 0.0
        return self.total_time / len(self.extractions)

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.extractions if e.success)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens or 0 for e in self.extractions)


# Global timing reports (one per extractor::model combination)
_timing_reports: dict[str, TimingReport] = {}


def get_timing_report(extractor: str, model: str) -> TimingReport:
    """Get or create the timing report for an extractor+model combination."""
    key = f"{extractor}::{model}"
    if key not in _timing_reports:
        _timing_reports[key] = TimingReport(key=key)
    return _timing_reports[key]


def write_stats_jsonl():
    """Append per-extraction stats to JSONL file for later analysis."""
    if not _timing_reports:
        return

    timestamp = datetime.now().isoformat()
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STATS_FILE, "a") as f:
        for report in _timing_reports.values():
            for extraction in report.extractions:
                record = {
                    "timestamp": timestamp,
                    "model": extraction.model,
                    "extractor": extraction.extractor,
                    "case_name": extraction.case_name,
                    "elapsed_seconds": extraction.elapsed_seconds,
                    "success": extraction.success,
                    "role_count": extraction.role_count,
                    "error_type": extraction.error_type,
                    "total_tokens": extraction.total_tokens,
                }
                f.write(json.dumps(record) + "\n")


def record_test_result(extractor: str, model: str, passed: bool):
    """Record a test pass/fail for an extractor+model combination."""
    report = get_timing_report(extractor, model)
    if passed:
        report.tests_passed += 1
    else:
        report.tests_failed += 1


def pytest_sessionfinish(session, exitstatus):
    """Override exit code - eval suites expect some failures."""
    # Always exit 0 for eval suite (failures are expected, not bugs)
    session.exitstatus = 0


def pytest_report_teststatus(report, config):
    """Customize test status display for eval suite."""
    if report.when == "call":
        if report.failed:
            return "failed", "✗", "MISS"
        elif report.passed:
            return "passed", "✓", "PASS"
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print timing summary at the end of test run."""
    if not _timing_reports:
        return

    if len(_timing_reports) > 1:
        # Print comparison table
        terminalreporter.write_sep("=", "COMPARISON")
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"{'Extractor::Model':<35} {'Avg Time':<12} {'Passed':<10} {'Total':<12} {'Tokens':<10}"
        )
        terminalreporter.write_line("-" * 79)

        for key in sorted(_timing_reports.keys()):
            report = _timing_reports[key]
            total_tests = report.tests_passed + report.tests_failed
            tokens_str = f"{report.total_tokens:,}" if report.total_tokens else "N/A"
            terminalreporter.write_line(
                f"{key:<35} {report.avg_time:>6.2f}s      "
                f"{report.tests_passed}/{total_tests:<7} "
                f"{report.total_time:>6.2f}s      "
                f"{tokens_str}"
            )

        terminalreporter.write_line("")
        terminalreporter.write_sep("=", "")
    else:
        # Single extractor+model summary
        for key, report in _timing_reports.items():
            if report.extractions:
                terminalreporter.write_sep("=", "TIMING SUMMARY")
                terminalreporter.write_line(f"Configuration: {key}")
                terminalreporter.write_line(
                    f"Total extractions: {len(report.extractions)}"
                )
                terminalreporter.write_line(f"Successful: {report.success_count}")
                terminalreporter.write_line(f"Total time: {report.total_time:.2f}s")
                terminalreporter.write_line(f"Average time: {report.avg_time:.2f}s")
                terminalreporter.write_line(f"Total tokens: {report.total_tokens:,}")
                terminalreporter.write_sep("=", "")

    # Persist stats to JSONL for later analysis
    write_stats_jsonl()
