"""Extract structured data from existing matches using LLM.

This script reads matches.json (or a subset) and adds extracted fields to each match.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Install with: pip install openai")
    sys.exit(1)

from who_is_hiring import extract_job_info_with_llm


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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
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
        extracted = extract_job_info_with_llm(full_content, client, matched_text=matched_text)
        match["extracted"] = extracted
        
        # Show what was extracted
        if extracted.get("extraction_error"):
            print(f"  ⚠️  Error: {extracted['extraction_error']}")
        else:
            company = extracted.get("company_name") or "N/A"
            role = extracted.get("role_name") or "N/A"
            remote = "Yes" if extracted.get("is_remote") else "No" if extracted.get("is_remote") is False else "N/A"
            comp = extracted.get("cash_compensation") or "N/A"
            print(f"  ✓ Company: {company}, Role: {role}, Remote: {remote}, Comp: {comp}")
        
        # Small delay to avoid rate limits
        time.sleep(0.1)
    
    # Write results
    print(f"\nWriting results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    
    # Summary
    successful = sum(1 for m in matches if m.get("extracted") and not m.get("extracted", {}).get("extraction_error"))
    print(f"\n✅ Complete! Successfully extracted data for {successful}/{len(matches)} matches")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract structured data from matches using LLM")
    parser.add_argument(
        "--input",
        default="matches.json",
        help="Input JSON file with matches (default: matches.json)"
    )
    parser.add_argument(
        "--output",
        default="matches_with_extraction.json",
        help="Output JSON file with extracted data (default: matches_with_extraction.json)"
    )
    
    args = parser.parse_args()
    extract_from_matches(args.input, args.output)

