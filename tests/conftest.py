"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest before running tests.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file before any tests run
# This ensures OPENAI_API_KEY is available for skip checks
load_dotenv()
