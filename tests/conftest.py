"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest before running tests.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file before any tests run
# This ensures API keys are available for skip checks
load_dotenv()

# Default model for tests
DEFAULT_TEST_MODEL = "gpt-4o-mini"


def pytest_addoption(parser):
    """Add custom command-line options for extraction tests."""
    parser.addoption(
        "--models",
        action="store",
        default=DEFAULT_TEST_MODEL,
        help=f"Comma-separated list of models to test (default: {DEFAULT_TEST_MODEL})",
    )


def get_test_models(config) -> list[str]:
    """Get the list of models to test based on CLI options."""
    models_str = config.getoption("--models")
    return [m.strip() for m in models_str.split(",")]


def pytest_configure(config):
    """Validate that the correct API keys are set."""
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


# --- TIMING INFRASTRUCTURE ---


@dataclass
class ExtractionTiming:
    """Tracks timing for a single extraction."""

    case_name: str
    model: str
    elapsed_seconds: float
    success: bool
    role_count: int = 0
    error_type: str | None = None


@dataclass
class TimingReport:
    """Aggregates timing data across all extractions."""

    model: str
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


# Global timing reports (one per model)
_timing_reports: dict[str, TimingReport] = {}


def get_timing_report(model: str) -> TimingReport:
    """Get or create the timing report for a model."""
    if model not in _timing_reports:
        _timing_reports[model] = TimingReport(model=model)
    return _timing_reports[model]


def record_test_result(model: str, passed: bool):
    """Record a test pass/fail for a model."""
    report = get_timing_report(model)
    if passed:
        report.tests_passed += 1
    else:
        report.tests_failed += 1


def infer_provider(model: str) -> str:
    """Infer provider from model name (duplicated from extract_jobs for independence)."""
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    elif model.startswith("gemini-"):
        return "gemini"
    return "openai"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print timing summary at the end of test run."""
    if not _timing_reports:
        return

    models = get_test_models(config)

    if len(_timing_reports) > 1:
        # Print comparison table
        terminalreporter.write_sep("=", "MODEL COMPARISON")
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"{'Model':<25} {'Avg Time':<12} {'Passed':<10} {'Total':<10}"
        )
        terminalreporter.write_line("-" * 57)

        for model in models:
            if model in _timing_reports:
                report = _timing_reports[model]
                total_tests = report.tests_passed + report.tests_failed
                terminalreporter.write_line(
                    f"{model:<25} {report.avg_time:>6.2f}s      "
                    f"{report.tests_passed}/{total_tests:<7} "
                    f"{report.total_time:>6.2f}s"
                )

        terminalreporter.write_line("")
        terminalreporter.write_sep("=", "")
    else:
        # Single model summary
        for model, report in _timing_reports.items():
            if report.extractions:
                terminalreporter.write_sep("=", "TIMING SUMMARY")
                terminalreporter.write_line(f"Model: {report.model}")
                terminalreporter.write_line(
                    f"Total extractions: {len(report.extractions)}"
                )
                terminalreporter.write_line(f"Successful: {report.success_count}")
                terminalreporter.write_line(f"Total time: {report.total_time:.2f}s")
                terminalreporter.write_line(f"Average time: {report.avg_time:.2f}s")
                terminalreporter.write_sep("=", "")
