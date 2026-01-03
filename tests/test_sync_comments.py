"""Test suite for sync_comments.py.

All tests use --post-id and --max-comments flags to avoid straining the API.
"""

import json
import os
import subprocess
from pathlib import Path

# Use a deterministic test post (Nov 2025 "Who is hiring?" thread)
TEST_POST_ID = "45800465"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_python_bin() -> Path:
    repo_root = get_repo_root()
    return Path(os.getenv("PYTHON_BIN", repo_root / ".venv" / "bin" / "python"))


def run_sync(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run sync_comments.py with given arguments."""
    repo_root = get_repo_root()
    python_bin = get_python_bin()

    result = subprocess.run(
        [str(python_bin), "sync_comments.py", *args],
        cwd=repo_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command failed: sync_comments.py {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def load_comments(path: Path) -> dict:
    """Load comments cache from file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestFetchPosts:
    """Test fetching posts functionality."""

    def test_fetch_specific_post(self, tmp_path: Path) -> None:
        """Fetching a specific post by ID works."""
        output = tmp_path / "comments.json"

        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "3",
                "--output",
                str(output),
            ]
        )

        assert output.exists()
        cache = load_comments(output)

        assert "items" in cache
        assert "metadata" in cache
        assert len(cache["items"]) <= 3
        assert cache["metadata"].get("schema_version") == 1
        assert cache["metadata"].get("last_synced_at") is not None

    def test_fetch_recent_posts(self, tmp_path: Path) -> None:
        """Fetching N recent posts works."""
        output = tmp_path / "comments.json"

        # Fetch just 1 post with 2 comments to minimize API calls
        run_sync(
            [
                "--posts",
                "1",
                "--max-comments",
                "2",
                "--output",
                str(output),
            ]
        )

        assert output.exists()
        cache = load_comments(output)
        assert "items" in cache
        assert len(cache["items"]) <= 2


class TestIncrementalSync:
    """Test incremental sync behavior."""

    def test_no_duplicates_on_resync(self, tmp_path: Path) -> None:
        """Running sync twice doesn't create duplicates."""
        output = tmp_path / "comments.json"

        # First sync
        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "3",
                "--output",
                str(output),
            ]
        )

        cache1 = load_comments(output)
        count1 = len(cache1["items"])
        ids1 = {c["id"] for c in cache1["items"] if c.get("id")}

        # Second sync (should be incremental, no new comments expected)
        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "3",
                "--output",
                str(output),
            ]
        )

        cache2 = load_comments(output)
        count2 = len(cache2["items"])
        ids2 = {c["id"] for c in cache2["items"] if c.get("id")}

        # Should have same comments (no duplicates)
        assert count1 == count2, f"Duplicate comments created: {count1} -> {count2}"
        assert ids1 == ids2

    def test_last_synced_at_updates(self, tmp_path: Path) -> None:
        """last_synced_at timestamp updates on each sync."""
        output = tmp_path / "comments.json"

        # First sync
        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "2",
                "--output",
                str(output),
            ]
        )

        cache1 = load_comments(output)
        ts1 = cache1["metadata"]["last_synced_at"]

        # Brief pause then second sync
        import time

        time.sleep(0.1)

        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "2",
                "--output",
                str(output),
            ]
        )

        cache2 = load_comments(output)
        ts2 = cache2["metadata"]["last_synced_at"]

        assert ts2 >= ts1, "last_synced_at should update on sync"


class TestTopLevelOnly:
    """Test that only top-level comments are fetched."""

    def test_comments_are_top_level(self, tmp_path: Path) -> None:
        """All fetched comments should be top-level (job postings, not replies)."""
        output = tmp_path / "comments.json"

        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "5",
                "--output",
                str(output),
            ]
        )

        cache = load_comments(output)
        comments = cache["items"]

        # All comments should have the post URL (indicating they're top-level)
        for c in comments:
            assert c.get("post_url"), f"Comment missing post_url: {c.get('id')}"
            assert TEST_POST_ID in c["post_url"], f"Comment has wrong post_url: {c}"


class TestCacheMerge:
    """Test cache merging behavior."""

    def test_existing_comments_preserved(self, tmp_path: Path) -> None:
        """Existing comments in cache are preserved when syncing."""
        output = tmp_path / "comments.json"

        # Create a fake existing cache with a synthetic comment
        fake_comment = {
            "id": "fake_12345",
            "post_url": "https://news.ycombinator.com/item?id=99999",
            "commenter": "test_user",
            "date": "2020-01-01T00:00:00+00:00",
            "content": "This is a fake comment for testing",
        }
        initial_cache = {
            "items": [fake_comment],
            "metadata": {
                "last_synced_at": "2020-01-01T00:00:00+00:00",
                "schema_version": 1,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(initial_cache, f)

        # Sync new comments
        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "2",
                "--output",
                str(output),
            ]
        )

        cache = load_comments(output)
        ids = {c["id"] for c in cache["items"]}

        # Fake comment should still be there
        assert "fake_12345" in ids, "Existing comments were not preserved"
        # New comments should also be there
        assert len(cache["items"]) >= 2


class TestRefreshFlag:
    """Test --refresh flag behavior."""

    def test_refresh_clears_cache(self, tmp_path: Path) -> None:
        """--refresh flag ignores existing cache and starts fresh."""
        output = tmp_path / "comments.json"

        # Create a cache with a fake comment
        fake_comment = {
            "id": "fake_99999",
            "post_url": "https://news.ycombinator.com/item?id=99999",
            "commenter": "old_user",
            "date": "2020-01-01T00:00:00+00:00",
            "content": "Old comment that should be removed",
        }
        initial_cache = {
            "items": [fake_comment],
            "metadata": {
                "last_synced_at": "2020-01-01T00:00:00+00:00",
                "schema_version": 1,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(initial_cache, f)

        # Sync with --refresh
        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "2",
                "--output",
                str(output),
                "--refresh",
            ]
        )

        cache = load_comments(output)
        ids = {c["id"] for c in cache["items"]}

        # Fake comment should be gone
        assert "fake_99999" not in ids, "--refresh did not clear existing cache"


class TestCommentSchema:
    """Test that comments have the expected schema."""

    def test_comment_fields(self, tmp_path: Path) -> None:
        """Comments have all required fields."""
        output = tmp_path / "comments.json"

        run_sync(
            [
                "--post-id",
                TEST_POST_ID,
                "--max-comments",
                "3",
                "--output",
                str(output),
            ]
        )

        cache = load_comments(output)

        for c in cache["items"]:
            assert "id" in c, "Comment missing 'id'"
            assert "post_url" in c, "Comment missing 'post_url'"
            assert "commenter" in c, "Comment missing 'commenter'"
            assert "date" in c, "Comment missing 'date'"
            assert "content" in c, "Comment missing 'content'"
