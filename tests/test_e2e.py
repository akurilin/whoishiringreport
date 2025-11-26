import os
import json
from pathlib import Path
import subprocess

import pytest
from dotenv import load_dotenv


def test_end_to_end_smoke(tmp_path: Path) -> None:
    # Load .env so the test uses the same OPENAI_API_KEY as the app.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for end-to-end extraction")

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(os.getenv("TEST_OUT", repo_root / "out" / "test"))
    out_dir.mkdir(parents=True, exist_ok=True)

    posts_path = out_dir / "posts.json"
    comments_path = out_dir / "comments.json"
    matches_path = out_dir / "matches.json"
    extracted_path = out_dir / "matches_with_extraction.json"
    report_path = out_dir / "report.html"

    python_bin = Path(os.getenv("PYTHON_BIN", repo_root / ".venv" / "bin" / "python"))

    def run_cmd(args):
        # Run the script as a black box via CLI.
        result = subprocess.run(
            [str(python_bin), "who_is_hiring.py", *args],
            cwd=repo_root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    # Use a deterministic test post with early UX/design hits (Nov 2025 thread).
    test_post_id = "45800465"

    # 1) Fetch posts (one only, or a specific ID).
    post_args = ["--output", str(posts_path), "--refresh-cache"]
    post_args.extend(["--post-id", str(test_post_id)])
    run_cmd(post_args)

    # 2) Fetch comments with caps.
    run_cmd(
        [
            "--fetch-comments",
            "--input",
            str(posts_path),
            "--output",
            str(comments_path),
            "--refresh-cache",
            "--comments-per-post",
            "10",
            "--max-comments-total",
            "10",
        ]
    )

    # 3) Search matches with extraction capped.
    run_cmd(
        [
            "--search-eng-management",
            "--profile",
            "profiles/ux_designer.yaml",
            "--input",
            str(comments_path),
            "--output",
            str(matches_path),
            "--max-matches",
            "10",
        ]
    )

    # 4) Extract (idempotent but forces the CLI path).
    run_cmd(
        [
            "--extract-from-matches",
            "--input",
            str(matches_path),
            "--output",
            str(extracted_path),
        ]
    )

    # 5) Generate HTML.
    run_cmd(
        [
            "--generate-html",
            "--input",
            str(extracted_path),
            "--output",
            str(report_path),
            "--no-open-report",
        ]
    )

    # Validations (black-box outputs).
    with open(posts_path, encoding="utf-8") as f:
        posts = json.load(f)
    assert len(posts) == 1, f"Expected 1 post, got {len(posts)}"

    with open(comments_path, encoding="utf-8") as f:
        comments = json.load(f)
    assert comments, "No comments fetched"
    assert len(comments) <= 10, f"Expected <=10 comments, got {len(comments)}"

    with open(matches_path, encoding="utf-8") as f:
        matches = json.load(f)
    assert matches, "No matches found; choose a thread with UX/design roles"
    assert len(matches) <= 10, f"Expected <=10 matches, got {len(matches)}"

    with open(extracted_path, encoding="utf-8") as f:
        extracted_matches = json.load(f)
    for i, match in enumerate(extracted_matches, 1):
        extracted = match.get("extracted") or {}
        assert extracted, f"Match {i} missing extraction payload"
        assert not extracted.get(
            "extraction_error"
        ), f"Match {i} has extraction_error: {extracted['extraction_error']}"

    assert report_path.is_file(), "Report was not generated"
    assert report_path.stat().st_size > 0, "Report is empty"
