"""Tests for tools/file_writer.py — safe file materialisation."""

import pytest
from pathlib import Path
from ai_cto.tools.file_writer import write_files, _safe_join, WORKSPACE_ROOT


def test_write_files_creates_files(tmp_path, monkeypatch):
    """Files should be written under workspace/<project_id>/."""
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)

    files = {
        "main.py": "print('hello')",
        "utils/helper.py": "def add(a, b): return a + b",
    }
    project_dir = write_files("test-proj", files)

    assert (project_dir / "main.py").read_text() == "print('hello')"
    assert (project_dir / "utils" / "helper.py").read_text() == "def add(a, b): return a + b"


def test_write_files_returns_project_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
    result = write_files("my-proj", {"a.py": "# hello"})
    assert result == tmp_path / "my-proj"


def test_safe_join_valid(tmp_path):
    result = _safe_join(tmp_path, "src/main.py")
    assert result == tmp_path / "src" / "main.py"


def test_safe_join_traversal_blocked(tmp_path):
    with pytest.raises(ValueError, match="Directory traversal"):
        _safe_join(tmp_path, "../../etc/passwd")
