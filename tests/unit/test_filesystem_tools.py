"""
Unit tests for the sandboxed filesystem tools — real filesystem operations
against a pytest tmp_path, not mocks, since the whole point of these tests
is verifying the sandbox boundary actually holds.
"""

from __future__ import annotations

import pytest

from orchestrator.tools.filesystem_tools import list_dir, read_file, write_file


@pytest.fixture(autouse=True)
def _sandbox_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path))
    yield tmp_path


def test_write_then_read_round_trip():
    write_file("notes.txt", "hello sandbox")

    assert read_file("notes.txt") == "hello sandbox"


def test_write_creates_parent_directories():
    write_file("nested/dir/file.txt", "content")

    assert read_file("nested/dir/file.txt") == "content"


def test_list_dir_reports_entries():
    write_file("a.txt", "x")
    write_file("sub/b.txt", "y")

    listing = list_dir(".")

    assert "a.txt" in listing
    assert "sub/" in listing


def test_read_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_file("does_not_exist.txt")


def test_path_traversal_is_rejected():
    with pytest.raises(ValueError):
        read_file("../../etc/passwd")


def test_absolute_path_outside_sandbox_is_rejected():
    with pytest.raises(ValueError):
        write_file("/tmp/outside_sandbox.txt", "nope")
