#!/usr/bin/env python3
"""Generate HTML report from extracted jobs data.

Usage:
    python generate_report.py                           # Use default paths
    python generate_report.py --input out/extracted_jobs.json --output out/report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import webbrowser
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

# --- CONSTANTS ---
BASE_DIR = Path(__file__).parent
DEFAULT_INPUT_PATH = BASE_DIR / "out" / "extracted_jobs.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "out" / "report.html"


# --- HTML SANITIZATION ---
class SafeHTMLParser(HTMLParser):
    """
    Sanitize HN comment HTML: allow only <a> tags (with safe hrefs), escape everything else.
    """

    ALLOWED_TAGS = {"a"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ALLOWED_TAGS:
            safe_attrs = []
            for name, value in attrs:
                if tag == "a" and name == "href" and value:
                    safe_href = html.escape(value, quote=True)
                    safe_attrs.append(f'href="{safe_href}" target="_blank"')
            if safe_attrs:
                self.parts.append(f"<{tag} {' '.join(safe_attrs)}>")
            else:
                self.parts.append(f"<{tag}>")
        else:
            self.parts.append(html.escape(self.get_starttag_text()))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ALLOWED_TAGS:
            self.parts.append(f"</{tag}>")
        else:
            self.parts.append(html.escape(f"</{tag}>"))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(html.escape(self.get_starttag_text()))

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))

    def get_safe_html(self) -> str:
        return "".join(self.parts)


def sanitize_html_content(raw_html: str) -> str:
    """Sanitize HTML content, keeping only safe <a> tags."""
    if not raw_html:
        return ""
    decoded = html.unescape(raw_html)
    parser = SafeHTMLParser()
    try:
        parser.feed(decoded)
        return parser.get_safe_html()
    except Exception:
        return html.escape(decoded)


def format_date(date_str: str | None) -> str:
    """Format ISO date string for display."""
    if not date_str:
        return ""
    try:
        dt_obj = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt_obj.strftime("%Y-%m-%d")
    except Exception:
        return date_str


def format_salary(item: dict) -> str:
    """Format salary range from extracted data."""
    salary_raw = item.get("salary_raw")
    if salary_raw:
        return salary_raw

    salary_min = item.get("salary_min")
    salary_max = item.get("salary_max")
    currency = item.get("salary_currency", "USD") or "USD"

    if salary_min and salary_max:
        if currency == "USD":
            return f"${salary_min // 1000}k-${salary_max // 1000}k"
        return f"{salary_min // 1000}k-{salary_max // 1000}k {currency}"
    elif salary_min:
        if currency == "USD":
            return f"${salary_min // 1000}k+"
        return f"{salary_min // 1000}k+ {currency}"
    elif salary_max:
        if currency == "USD":
            return f"Up to ${salary_max // 1000}k"
        return f"Up to {salary_max // 1000}k {currency}"

    return ""


def generate_html_report(
    input_path: Path,
    output_path: Path,
    open_browser: bool = True,
) -> None:
    """Generate HTML report from extracted jobs JSON.

    Args:
        input_path: Path to extracted_jobs.json
        output_path: Path to write output HTML file
        open_browser: Whether to open the report in the default browser
    """
    print(f"Loading extracted jobs from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    # Filter out error rows and non-job postings
    jobs = [
        item for item in items if item.get("is_job_posting") and item.get("role_title")
    ]

    print(f"Loaded {len(jobs)} job roles")
    print("Generating HTML report...")

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(jobs):
        is_remote = item.get("is_remote")
        locations = item.get("locations", [])
        raw_content = item.get("raw_content", "") or ""

        # Only include application_method if it looks like a URL
        apply_method = (item.get("application_method") or "").strip()
        apply_url = (
            apply_method if apply_method.startswith(("http://", "https://")) else ""
        )

        rows.append(
            {
                "row_id": f"row-{idx}",
                "company": (item.get("company_name") or "").strip(),
                "company_url": (item.get("company_url") or "").strip(),
                "company_stage": (item.get("company_stage") or "").strip(),
                "role": (item.get("role_title") or "").strip(),
                "location": ", ".join(locations) if locations else "",
                "remote": "Yes" if is_remote else "No" if is_remote is False else "—",
                "remote_class": (
                    "remote-yes"
                    if is_remote
                    else "remote-no"
                    if is_remote is False
                    else ""
                ),
                "employment": (item.get("employment_type") or "").strip(),
                "cash_comp": format_salary(item),
                "equity": item.get("equity") or "—",
                "apply_url": apply_url,
                "commenter": (item.get("commenter") or "").strip(),
                "date": format_date(item.get("comment_date")),
                "post_url": item.get("post_url", "") or "",
                "full_content_html": sanitize_html_content(raw_content),
            }
        )

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    html_content = template.render(rows=rows, total_matches=len(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated: {output_path}")

    if open_browser:
        print("Opening in browser...")
        webbrowser.open(f"file://{output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from extracted jobs data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON file (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output HTML file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the report in browser",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run extract_jobs.py first to generate the extracted jobs data.")
        return 1

    generate_html_report(
        input_path=args.input,
        output_path=args.output,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    exit(main())
