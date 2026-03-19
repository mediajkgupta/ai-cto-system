"""
test_sandbox_runtime.py — Unit tests for sandbox.py RuntimeConfig and
stack-aware execution helpers.

These tests do NOT require Docker or a running subprocess — they only
test the configuration / selection logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from ai_cto.tools.sandbox import get_runtime_config, docker_available


# ── get_runtime_config ──────────────────────────────────────────────────────────

class TestGetRuntimeConfigPython:
    def test_empty_files_defaults_to_python(self):
        cfg = get_runtime_config({}, "main.py")
        assert cfg["image"] == "python:3.11-slim"
        assert cfg["command"] == ["python", "main.py"]

    def test_python_without_requirements(self):
        files = {"main.py": "print('hello')"}
        cfg = get_runtime_config(files, "main.py")
        assert cfg["install_cmd"] is None

    def test_python_with_requirements(self):
        files = {"main.py": "import flask", "requirements.txt": "flask"}
        cfg = get_runtime_config(files, "main.py")
        assert cfg["install_cmd"] == ["pip", "install", "-q", "-r", "requirements.txt"]

    def test_python_memory_limit(self):
        cfg = get_runtime_config({}, "main.py")
        assert cfg["memory"] == "256m"

    def test_entry_point_passed_through(self):
        cfg = get_runtime_config({}, "app.py")
        assert cfg["command"] == ["python", "app.py"]


class TestGetRuntimeConfigNode:
    def test_js_file_selects_node(self):
        files = {"server.js": "const x = 1;"}
        cfg = get_runtime_config(files, "server.js")
        assert cfg["image"] == "node:18-alpine"
        assert cfg["command"] == ["node", "server.js"]

    def test_ts_file_selects_node(self):
        files = {"index.ts": "export {}"}
        cfg = get_runtime_config(files, "index.ts")
        assert cfg["image"] == "node:18-alpine"

    def test_package_json_selects_node(self):
        files = {"package.json": '{"name":"app"}'}
        cfg = get_runtime_config(files, "index.js")
        assert cfg["image"] == "node:18-alpine"

    def test_node_with_package_json_has_install_cmd(self):
        files = {"server.js": "const x=1;", "package.json": "{}"}
        cfg = get_runtime_config(files, "server.js")
        assert cfg["install_cmd"] == ["npm", "install", "--production"]

    def test_node_without_package_json_no_install(self):
        files = {"server.js": "const x=1;"}
        cfg = get_runtime_config(files, "server.js")
        assert cfg["install_cmd"] is None

    def test_node_memory_limit(self):
        files = {"app.js": ""}
        cfg = get_runtime_config(files, "app.js")
        assert cfg["memory"] == "512m"

    def test_node_entry_point_passed_through(self):
        files = {"app.js": "const x=1;"}
        cfg = get_runtime_config(files, "app.js")
        assert cfg["command"] == ["node", "app.js"]


# ── docker_available ────────────────────────────────────────────────────────────

class TestDockerAvailable:
    def test_returns_false_when_docker_not_found(self):
        with patch("shutil.which", return_value=None):
            assert docker_available() is False

    def test_returns_false_when_daemon_unreachable(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                assert docker_available() is False

    def test_returns_true_when_docker_running(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert docker_available() is True

    def test_returns_false_on_timeout(self):
        import subprocess
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 5)):
                assert docker_available() is False
