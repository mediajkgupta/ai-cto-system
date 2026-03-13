"""
Tests for verification/checks.py — all five static checkers.

No LLM calls, no filesystem I/O. Every test exercises checkers directly
against small inline code snippets.
"""

import pytest
from ai_cto.verification.checks import (
    SyntaxChecker,
    SecurityChecker,
    DependencyChecker,
    ArchitectureChecker,
    RuntimeRiskChecker,
    run_all_checks,
    VerificationResult,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def issues_of(result_list, severity=None, check=None):
    issues = result_list
    if severity:
        issues = [i for i in issues if i["severity"] == severity]
    if check:
        issues = [i for i in issues if i["check"] == check]
    return issues


# ── SyntaxChecker ──────────────────────────────────────────────────────────────

class TestSyntaxChecker:
    checker = SyntaxChecker()

    def test_valid_python_no_issues(self):
        files = {"main.py": "def hello():\n    return 'hello'\n"}
        assert self.checker.check(files, "main.py") == []

    def test_syntax_error_reported_as_error(self):
        files = {"main.py": "def hello(\n    pass\n"}
        issues = self.checker.check(files, "main.py")
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["check"] == "syntax"
        assert issues[0]["file"] == "main.py"

    def test_syntax_error_has_line_number(self):
        files = {"main.py": "x = 1\ny = (\nz = 3\n"}
        issues = self.checker.check(files, "main.py")
        assert issues[0]["line"] is not None

    def test_non_python_files_ignored(self):
        files = {"README.md": "# hello", "config.yaml": "key: value"}
        assert self.checker.check(files, "main.py") == []

    def test_multiple_files_each_checked(self):
        files = {
            "good.py": "x = 1",
            "bad.py": "def broken(\n    pass",
        }
        issues = self.checker.check(files, "good.py")
        assert any(i["file"] == "bad.py" for i in issues)
        assert not any(i["file"] == "good.py" for i in issues)


# ── SecurityChecker ────────────────────────────────────────────────────────────

class TestSecurityChecker:
    checker = SecurityChecker()

    def test_eval_flagged(self):
        files = {"main.py": "result = eval(user_input)\n"}
        issues = self.checker.check(files, "main.py")
        assert any("eval" in i["message"] for i in issues)

    def test_exec_flagged(self):
        files = {"main.py": "exec(code)\n"}
        issues = self.checker.check(files, "main.py")
        assert any("exec" in i["message"] for i in issues)

    def test_os_system_flagged(self):
        files = {"main.py": "import os\nos.system('rm -rf /')\n"}
        issues = self.checker.check(files, "main.py")
        assert any("os.system" in i["message"] for i in issues)

    def test_hardcoded_secret_flagged(self):
        files = {"config.py": "api_key = 'sk-abcdefghijklmnop'\n"}
        issues = self.checker.check(files, "config.py")
        assert any("hardcoded secret" in i["message"].lower() for i in issues)

    def test_path_traversal_is_error(self):
        files = {"main.py": 'path = "../../../etc/passwd"\n'}
        issues = self.checker.check(files, "main.py")
        errors = [i for i in issues if i["severity"] == "error"]
        assert any("traversal" in i["message"].lower() for i in errors)

    def test_clean_code_no_issues(self):
        files = {"main.py": "def add(a, b):\n    return a + b\n"}
        assert self.checker.check(files, "main.py") == []

    def test_syntax_error_file_skipped_gracefully(self):
        """SecurityChecker must not raise when a file has a syntax error."""
        files = {"bad.py": "def broken(\n    pass"}
        # Should not raise
        issues = self.checker.check(files, "bad.py")
        assert isinstance(issues, list)


# ── DependencyChecker ──────────────────────────────────────────────────────────

class TestDependencyChecker:
    checker = DependencyChecker()

    def test_stdlib_imports_not_flagged(self):
        files = {"main.py": "import os\nimport sys\nfrom pathlib import Path\n"}
        issues = self.checker.check(files, "main.py")
        assert issues == []

    def test_missing_third_party_flagged(self):
        files = {"main.py": "import requests\n"}
        issues = self.checker.check(files, "main.py")
        assert any("requests" in i["message"] for i in issues)
        assert all(i["severity"] == "warning" for i in issues)

    def test_declared_requirement_not_flagged(self):
        files = {
            "main.py": "import requests\n",
            "requirements.txt": "requests>=2.28.0\n",
        }
        issues = self.checker.check(files, "main.py")
        assert issues == []

    def test_requirement_with_specifier_parsed_correctly(self):
        files = {
            "main.py": "import flask\n",
            "requirements.txt": "flask>=2.0,<3.0\n",
        }
        issues = self.checker.check(files, "main.py")
        assert issues == []

    def test_relative_imports_not_flagged(self):
        files = {"pkg/mod.py": "from . import helper\nfrom .utils import fn\n"}
        issues = self.checker.check(files, "pkg/mod.py")
        assert issues == []

    def test_intra_project_imports_not_flagged(self):
        files = {
            "main.py": "from utils import helper\n",
            "utils.py": "def helper(): pass\n",
        }
        issues = self.checker.check(files, "main.py")
        assert issues == []


# ── ArchitectureChecker ────────────────────────────────────────────────────────

class TestArchitectureChecker:
    checker = ArchitectureChecker()

    def test_missing_entry_point_is_error(self):
        files = {"utils.py": "def add(a, b): return a+b\n"}
        issues = self.checker.check(files, "main.py")
        errors = [i for i in issues if i["severity"] == "error"]
        assert any("Entry point" in i["message"] for i in errors)

    def test_present_entry_point_no_error(self):
        files = {"main.py": "print('hello')\n"}
        issues = self.checker.check(files, "main.py")
        assert not any("Entry point" in i["message"] for i in issues)

    def test_invalid_path_characters_flagged(self):
        files = {"my file.py": "x = 1\n"}   # space in path
        issues = self.checker.check(files, "my file.py")
        assert any("unusual characters" in i["message"] for i in issues)

    def test_circular_import_detected(self):
        files = {
            "a.py": "from b import something\n",
            "b.py": "from a import something\n",
        }
        issues = self.checker.check(files, "a.py")
        assert any("circular" in i["message"].lower() for i in issues)

    def test_no_circular_import_in_linear_chain(self):
        files = {
            "main.py": "from utils import helper\n",
            "utils.py": "def helper(): pass\n",
        }
        issues = self.checker.check(files, "main.py")
        assert not any("circular" in i["message"].lower() for i in issues)


# ── RuntimeRiskChecker ─────────────────────────────────────────────────────────

class TestRuntimeRiskChecker:
    checker = RuntimeRiskChecker()

    def test_infinite_loop_flagged(self):
        code = "while True:\n    do_something()\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        assert any("infinite loop" in i["message"].lower() for i in issues)

    def test_while_true_with_break_ok(self):
        code = "while True:\n    x = read()\n    if x: break\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        assert not any("infinite loop" in i["message"].lower() for i in issues)

    def test_conditional_while_not_flagged(self):
        code = "while queue:\n    process(queue.pop())\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        assert not any("infinite loop" in i["message"].lower() for i in issues)

    def test_recursion_without_base_case_flagged(self):
        code = "def recurse(n):\n    return recurse(n - 1)\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        assert any("base case" in i["message"].lower() for i in issues)

    def test_recursion_with_base_case_ok(self):
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        issues = self.checker.check({"main.py": code}, "main.py")
        assert not any("base case" in i["message"].lower() for i in issues)

    def test_divide_by_zero_literal_is_error(self):
        code = "result = 10 / 0\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        errors = [i for i in issues if i["severity"] == "error"]
        assert any("zero" in i["message"].lower() for i in errors)

    def test_safe_division_not_flagged(self):
        code = "result = 10 / divisor\n"
        issues = self.checker.check({"main.py": code}, "main.py")
        assert not any("zero" in i["message"].lower() for i in issues)


# ── run_all_checks orchestrator ────────────────────────────────────────────────

class TestRunAllChecks:

    def test_clean_code_passes(self):
        files = {"main.py": "def add(a, b):\n    return a + b\n\nprint(add(1,2))\n"}
        result = run_all_checks(files, "main.py")
        assert result["passed"] is True
        assert result["error_count"] == 0

    def test_syntax_error_fails(self):
        files = {"main.py": "def broken(\n    pass\n"}
        result = run_all_checks(files, "main.py")
        assert result["passed"] is False
        assert result["error_count"] >= 1

    def test_result_has_required_keys(self):
        files = {"main.py": "x = 1\n"}
        result = run_all_checks(files, "main.py")
        for key in ("passed", "issues", "generated_tests", "error_count", "warning_count", "summary"):
            assert key in result

    def test_generated_tests_empty_by_default(self):
        files = {"main.py": "x = 1\n"}
        result = run_all_checks(files, "main.py")
        assert result["generated_tests"] == {}

    def test_summary_is_non_empty_string(self):
        files = {"main.py": "x = 1\n"}
        result = run_all_checks(files, "main.py")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_multiple_checkers_aggregate_issues(self):
        files = {
            "main.py": (
                "import requests\n"           # dep warning
                "result = eval(user_input)\n" # security warning
                "x = 10 / 0\n"               # runtime error
            )
        }
        result = run_all_checks(files, "main.py")
        checks_found = {i["check"] for i in result["issues"]}
        assert "security" in checks_found
        assert "dependency" in checks_found
        assert "runtime" in checks_found
