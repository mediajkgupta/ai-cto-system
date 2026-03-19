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
    EmptyEntryPointChecker,
    NodeJsChecker,
    detect_stack,
    detect_node_entry,
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


# ── detect_stack() ─────────────────────────────────────────────────────────────

class TestDetectStack:
    def test_js_files_detected_as_node(self):
        assert detect_stack({"server.js": "const x = 1"}) == "node"

    def test_package_json_detected_as_node(self):
        assert detect_stack({"package.json": "{}"}) == "node"

    def test_py_files_detected_as_python(self):
        assert detect_stack({"app.py": "x = 1"}) == "python"

    def test_empty_main_py_only_is_unknown(self):
        # An empty main.py (DebugAgent placeholder) must NOT count as Python
        assert detect_stack({"main.py": ""}) == "unknown"
        assert detect_stack({"main.py": "   \n  "}) == "unknown"

    def test_js_and_py_mixed_detected_as_node(self):
        # JS presence wins over Python
        assert detect_stack({"server.js": "x", "main.py": ""}) == "node"

    def test_empty_files_dict_is_unknown(self):
        assert detect_stack({}) == "unknown"


# ── detect_node_entry() ────────────────────────────────────────────────────────

class TestDetectNodeEntry:
    def test_server_js_preferred(self):
        files = {"server.js": "x", "app.js": "y", "index.js": "z"}
        assert detect_node_entry(files) == "server.js"

    def test_app_js_when_no_server(self):
        assert detect_node_entry({"app.js": "x"}) == "app.js"

    def test_empty_server_js_skipped(self):
        files = {"server.js": "", "app.js": "x"}
        assert detect_node_entry(files) == "app.js"

    def test_returns_none_when_all_empty(self):
        assert detect_node_entry({"server.js": "", "app.js": "  "}) is None

    def test_returns_none_for_empty_files(self):
        assert detect_node_entry({}) is None

    def test_falls_back_to_any_nonempty_js(self):
        files = {"routes/api.js": "module.exports = {}"}
        assert detect_node_entry(files) == "routes/api.js"


# ── EmptyEntryPointChecker ─────────────────────────────────────────────────────

class TestEmptyEntryPointChecker:
    checker = EmptyEntryPointChecker()

    def test_empty_entry_is_error(self):
        files = {"main.py": ""}
        issues = self.checker.check(files, "main.py")
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["check"] == "empty_entry"

    def test_whitespace_only_entry_is_error(self):
        files = {"main.py": "   \n\t\n  "}
        issues = self.checker.check(files, "main.py")
        assert len(issues) == 1

    def test_nonempty_entry_passes(self):
        files = {"main.py": "x = 1\n"}
        assert self.checker.check(files, "main.py") == []

    def test_missing_entry_produces_no_issues(self):
        # ArchitectureChecker handles missing entry; this checker only handles empty
        assert self.checker.check({}, "main.py") == []

    def test_empty_js_entry_is_error(self):
        files = {"server.js": ""}
        issues = self.checker.check(files, "server.js")
        assert len(issues) == 1
        assert "empty" in issues[0]["message"].lower()


# ── NodeJsChecker ──────────────────────────────────────────────────────────────

_VALID_SERVER = "const express = require('express');\nconst app = express();\nmodule.exports = app;\n"
_VALID_MODEL  = "const mongoose = require('mongoose');\nconst s = new mongoose.Schema({title: String});\nmodule.exports = mongoose.model('T', s);\n"
_VALID_ROUTES = "const express = require('express');\nconst router = express.Router();\nrouter.get('/', (req, res) => res.json([]));\nmodule.exports = router;\n"


class TestNodeJsCheckerNotNodeProject:
    checker = NodeJsChecker()

    def test_python_project_not_checked(self):
        """NodeJsChecker must be silent on pure Python projects."""
        files = {"app.py": "import flask\napp = flask.Flask(__name__)\n"}
        assert self.checker.check(files, "app.py") == []

    def test_empty_files_not_checked(self):
        assert self.checker.check({}, "main.py") == []


class TestNodeJsCheckerEntryPoint:
    checker = NodeJsChecker()

    def test_no_entry_file_is_error(self):
        # All JS files present but all empty
        files = {"server.js": "", "app.js": "  "}
        issues = self.checker.check(files, "server.js")
        errors = [i for i in issues if i["check"] == "nodejs" and i["severity"] == "error"]
        assert any("no non-empty entry" in i["message"].lower() for i in errors)

    def test_valid_server_js_passes_entry_check(self):
        files = {"server.js": _VALID_SERVER}
        issues = self.checker.check(files, "server.js")
        entry_errors = [i for i in issues if "no non-empty entry" in i["message"].lower()]
        assert entry_errors == []


class TestNodeJsCheckerEmptyFiles:
    checker = NodeJsChecker()

    def test_empty_js_file_is_error(self):
        files = {"server.js": _VALID_SERVER, "db.js": ""}
        issues = self.checker.check(files, "server.js")
        empty_errors = [i for i in issues if "empty" in i["message"].lower() and i["file"] == "db.js"]
        assert len(empty_errors) == 1
        assert empty_errors[0]["severity"] == "error"

    def test_nonempty_js_files_no_empty_errors(self):
        files = {"server.js": _VALID_SERVER, "db.js": "module.exports = {};\n"}
        issues = self.checker.check(files, "server.js")
        empty_errors = [i for i in issues if "empty" in i["message"].lower()]
        assert empty_errors == []


class TestNodeJsCheckerSyntax:
    checker = NodeJsChecker()

    def test_valid_js_passes_syntax(self):
        files = {"server.js": _VALID_SERVER}
        issues = self.checker.check(files, "server.js")
        syntax_errors = [i for i in issues if "SyntaxError" in i["message"]]
        assert syntax_errors == []

    def test_invalid_js_syntax_is_error(self):
        bad_js = "const x = {a: 1\nb: 2};\n"   # missing comma
        files = {"server.js": bad_js}
        issues = self.checker.check(files, "server.js")
        syntax_errors = [i for i in issues if "SyntaxError" in i["message"]]
        assert len(syntax_errors) == 1
        assert syntax_errors[0]["severity"] == "error"
        assert syntax_errors[0]["file"] == "server.js"

    def test_schema_missing_comma_is_syntax_error(self):
        """Exact pattern from the real Ollama run: missing comma in Mongoose schema."""
        bad_model = (
            "const mongoose = require('mongoose');\n"
            "const taskSchema = new mongoose.Schema({\n"
            "  title: { type: String, required: true },\n"
            "  status: { type: String }\n"
            "  createdAt: { type: Date }\n"   # <-- missing comma before createdAt
            "});\n"
            "module.exports = mongoose.model('Task', taskSchema);\n"
        )
        files = {"models/task.js": bad_model}
        issues = self.checker.check(files, "server.js")
        syntax_errors = [i for i in issues if "SyntaxError" in i["message"]]
        assert len(syntax_errors) == 1
        assert syntax_errors[0]["file"] == "models/task.js"

    def test_syntax_error_includes_line_number(self):
        bad_js = "const x = 1;\nconst y = {a: 1\nb: 2};\n"
        files = {"server.js": bad_js}
        issues = self.checker.check(files, "server.js")
        syntax_errors = [i for i in issues if "SyntaxError" in i["message"]]
        assert syntax_errors[0]["line"] is not None


class TestNodeJsCheckerRequireTargets:
    checker = NodeJsChecker()

    def test_missing_local_require_is_error(self):
        """require('../controllers/task') when controllers/task.js doesn't exist."""
        files = {
            "server.js": _VALID_SERVER,
            "api/tasks.js": "const T = require('../controllers/task');\n",
        }
        issues = self.checker.check(files, "server.js")
        req_errors = [i for i in issues if "not found in generated files" in i["message"]]
        assert len(req_errors) == 1
        assert req_errors[0]["file"] == "api/tasks.js"
        assert req_errors[0]["severity"] == "error"

    def test_present_local_require_passes(self):
        files = {
            "server.js": _VALID_SERVER,
            "api/tasks.js": "const T = require('../controllers/task');\n",
            "controllers/task.js": "module.exports = {};\n",
        }
        issues = self.checker.check(files, "server.js")
        req_errors = [i for i in issues if "not found in generated files" in i["message"]]
        assert req_errors == []

    def test_require_without_dot_not_checked(self):
        """Non-local require('mongoose') must not trigger file-existence check."""
        files = {"server.js": "const m = require('mongoose');\n"}
        issues = self.checker.check(files, "server.js")
        req_errors = [i for i in issues if "not found in generated files" in i["message"]]
        assert req_errors == []

    def test_require_js_extension_optional(self):
        """require('./db') should resolve to db.js."""
        files = {
            "server.js": "const db = require('./db');\n",
            "db.js": "module.exports = {};\n",
        }
        issues = self.checker.check(files, "server.js")
        req_errors = [i for i in issues if "not found in generated files" in i["message"]]
        assert req_errors == []


# ── Full pipeline false-positive regression test ────────────────────────────────

class TestNodeJsFalsePositiveRegression:
    """
    Reproduces the exact file set from the failing Ollama run and verifies the
    new verifier catches all issues that caused the false PASS.
    """

    def _ollama_run_files(self):
        return {
            # server.js — empty (0 bytes, as produced)
            "server.js": "",
            # main.py — empty placeholder created by DebugAgent
            "main.py": "",
            # db.js — valid connection string
            "db.js": (
                "\nconst mongoose = require('mongoose');\n"
                "mongoose.connect(process.env.MONGODB_URL || 'mongodb://localhost:27017/task-tracker', "
                "{ useNewUrlParser: true, useUnifiedTopology: true });\n"
            ),
            # models/task.js — missing comma syntax error
            "models/task.js": (
                "\nconst mongoose = require('mongoose');\n"
                "\nconst taskSchema = new mongoose.Schema({\n"
                "title: {\n  type: String,\n  required: true\n},\n"
                "description: {\n  type: String,\n},\n"
                "status: {\n  type: String,\n  enum: ['Not Started', 'In Progress', 'Completed'],\n  default: 'Not Started'\n}\n"
                "createdAt: {\n  type: Date,\n  default: Date.now\n}\n"
                "});\n\nmodule.exports = mongoose.model('Task', taskSchema);\n"
            ),
            # api/tasks.js — requires controllers/task which doesn't exist
            "api/tasks.js": (
                "\nconst express = require('express');\n"
                "const router = express.Router();\n"
                "const TaskController = require('../controllers/task');\n"
                "router.get('/', TaskController.getTasks);\n"
                "exports = module.exports = router;\n"
            ),
        }

    def test_stack_detected_as_node(self):
        assert detect_stack(self._ollama_run_files()) == "node"

    def test_node_entry_is_db_js_not_server(self):
        """server.js is empty, so detect_node_entry skips it and falls back."""
        entry = detect_node_entry(self._ollama_run_files())
        assert entry != "server.js"   # server.js is empty
        assert entry is not None       # something non-empty found

    def test_run_all_checks_fails(self):
        """The false-positive PASS must no longer occur."""
        files = self._ollama_run_files()
        result = run_all_checks(files, "server.js")
        assert result["passed"] is False

    def test_empty_server_js_caught(self):
        files = self._ollama_run_files()
        result = run_all_checks(files, "server.js")
        empty_errors = [
            i for i in result["issues"]
            if i["severity"] == "error" and "empty" in i["message"].lower()
            and i["file"] == "server.js"
        ]
        assert len(empty_errors) >= 1

    def test_syntax_error_in_model_caught(self):
        files = self._ollama_run_files()
        result = run_all_checks(files, "server.js")
        syntax_errors = [
            i for i in result["issues"]
            if i["severity"] == "error" and "SyntaxError" in i["message"]
            and "task.js" in i["file"]
        ]
        assert len(syntax_errors) >= 1

    def test_missing_controllers_task_caught(self):
        files = self._ollama_run_files()
        result = run_all_checks(files, "server.js")
        missing_errors = [
            i for i in result["issues"]
            if i["severity"] == "error" and "controllers/task" in i["message"]
        ]
        assert len(missing_errors) >= 1

    def test_empty_main_py_caught(self):
        files = self._ollama_run_files()
        result = run_all_checks(files, "main.py")
        empty_errors = [
            i for i in result["issues"]
            if i["severity"] == "error" and "empty" in i["message"].lower()
            and i["file"] == "main.py"
        ]
        assert len(empty_errors) >= 1
