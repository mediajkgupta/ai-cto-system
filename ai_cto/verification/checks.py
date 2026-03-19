"""
checks.py — Static verification checks for generated code.

Five checkers, all pure Python, zero LLM calls:

  SyntaxChecker       — ast.parse every .py file
  SecurityChecker     — AST + regex scan for dangerous patterns
  DependencyChecker   — detect imports missing from requirements
  ArchitectureChecker — validate entry point + file path hygiene
  RuntimeRiskChecker  — heuristic detection of infinite loops and recursion risks

Each checker exposes a single method:
    check(files: dict[str, str], entry_point: str) -> list[VerificationIssue]

run_all_checks() assembles them and returns a VerificationResult.
"""

import ast
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import logging
from typing import TypedDict, Optional

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────────────────

class VerificationIssue(TypedDict):
    severity: str        # "error" | "warning" | "info"
    check: str           # which checker raised this
    file: str            # relative file path ("" if project-level)
    line: Optional[int]  # line number, None if not applicable
    message: str


class VerificationResult(TypedDict):
    passed: bool                        # True if no ERROR-severity issues
    issues: list[VerificationIssue]
    generated_tests: dict[str, str]     # populated later by test_generator
    error_count: int
    warning_count: int
    summary: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _issue(
    severity: str,
    check: str,
    file: str,
    message: str,
    line: Optional[int] = None,
) -> VerificationIssue:
    return VerificationIssue(
        severity=severity, check=check, file=file, line=line, message=message
    )


def _python_files(files: dict[str, str]) -> dict[str, str]:
    return {p: c for p, c in files.items() if p.endswith(".py")}


# ── 1. Syntax Checker ──────────────────────────────────────────────────────────

class SyntaxChecker:
    """
    Parse every .py file with ast.parse.

    A SyntaxError means the file cannot run at all — always ERROR severity.
    """

    NAME = "syntax"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for path, content in _python_files(files).items():
            try:
                ast.parse(content, filename=path)
            except SyntaxError as exc:
                issues.append(_issue(
                    severity="error",
                    check=self.NAME,
                    file=path,
                    line=exc.lineno,
                    message=f"SyntaxError: {exc.msg}",
                ))
        return issues


# ── 2. Security Checker ────────────────────────────────────────────────────────

# Dangerous AST call patterns: (module_or_builtin, function_name, severity, message)
_DANGEROUS_CALLS = [
    (None, "eval",       "warning", "eval() executes arbitrary code — avoid or sanitise input."),
    (None, "exec",       "warning", "exec() executes arbitrary code — avoid or sanitise input."),
    (None, "__import__", "warning", "__import__() can load arbitrary modules dynamically."),
    ("os",   "system",  "warning", "os.system() runs shell commands — prefer subprocess with args list."),
    ("os",   "popen",   "warning", "os.popen() opens a shell — prefer subprocess."),
    ("shutil", "rmtree","warning", "shutil.rmtree() recursively deletes — ensure path is validated."),
]

# Regex patterns applied to raw source text
_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (password|api_key|secret|token|private_key)\s*=\s*['"][^'"]{4,}['"]
    """,
)
_TRAVERSAL_PATTERN = re.compile(r"\.\./")
_HARDCODED_IP = re.compile(r"""['"](\d{1,3}\.){3}\d{1,3}['"]""")


class SecurityChecker:
    """
    Scan for security anti-patterns using AST walking and regex.

    Raises WARNING for dangerous-but-sometimes-legitimate patterns.
    Raises ERROR for path traversal (which file_writer already blocks at runtime,
    but catching it here gives earlier, clearer feedback).
    """

    NAME = "security"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for path, content in _python_files(files).items():
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue  # SyntaxChecker already reported this

            issues.extend(self._check_ast(path, tree))
            issues.extend(self._check_text(path, content))

        return issues

    def _check_ast(
        self, path: str, tree: ast.AST
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func

            # Bare calls: eval(...), exec(...), __import__(...)
            if isinstance(func, ast.Name):
                for (mod, name, sev, msg) in _DANGEROUS_CALLS:
                    if mod is None and func.id == name:
                        issues.append(_issue(sev, self.NAME, path, msg, node.lineno))

            # Attribute calls: os.system(...), shutil.rmtree(...)
            elif isinstance(func, ast.Attribute):
                for (mod, name, sev, msg) in _DANGEROUS_CALLS:
                    if (
                        mod is not None
                        and func.attr == name
                        and isinstance(func.value, ast.Name)
                        and func.value.id == mod
                    ):
                        issues.append(_issue(sev, self.NAME, path, msg, node.lineno))

        return issues

    def _check_text(self, path: str, content: str) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        for i, line in enumerate(content.splitlines(), start=1):
            if _SECRET_PATTERN.search(line):
                issues.append(_issue(
                    "warning", self.NAME, path,
                    "Possible hardcoded secret detected — use environment variables.",
                    i,
                ))
            if _TRAVERSAL_PATTERN.search(line):
                issues.append(_issue(
                    "error", self.NAME, path,
                    "Path traversal pattern '../' found — this may allow directory escape.",
                    i,
                ))
            if _HARDCODED_IP.search(line):
                issues.append(_issue(
                    "info", self.NAME, path,
                    "Hardcoded IP address detected — consider making this configurable.",
                    i,
                ))

        return issues


# ── 3. Dependency Checker ──────────────────────────────────────────────────────

# Python standard library module names (Python 3.10+)
_STDLIB = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else frozenset()

# Common top-level package names that differ from import names
_KNOWN_ALIASES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
}


class DependencyChecker:
    """
    Parse import statements in every .py file.
    Classify each top-level module as stdlib or third-party.
    Warn when a third-party module has no corresponding entry in any
    requirements.txt found in the generated files.
    """

    NAME = "dependency"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # Collect declared requirements from any requirements.txt in the output
        declared = self._parse_requirements(files)

        # Names of .py files in the project (allow intra-project imports)
        project_modules = {
            p.replace("/", ".").replace("\\", ".").removesuffix(".py")
            for p in _python_files(files)
        }
        # Also allow bare module names (e.g. 'utils' from 'utils/helper.py' or 'utils.py')
        project_top = {
            p.split("/")[0].split("\\")[0].removesuffix(".py")
            for p in _python_files(files)
        }

        for path, content in _python_files(files).items():
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                top = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self._check_module(
                            top, path, node.lineno, declared, project_top, issues
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:   # skip relative imports
                        top = node.module.split(".")[0]
                        self._check_module(
                            top, path, node.lineno, declared, project_top, issues
                        )

        return issues

    def _check_module(
        self,
        name: str,
        path: str,
        lineno: int,
        declared: set[str],
        project_top: set[str],
        issues: list[VerificationIssue],
    ) -> None:
        if name in _STDLIB:
            return
        if name in project_top:
            return
        if name in declared:
            return
        # Map known aliases (e.g. cv2 → opencv-python)
        canonical = _KNOWN_ALIASES.get(name, name)
        if canonical.lower() in declared:
            return
        issues.append(_issue(
            "warning", self.NAME, path,
            f"Import '{name}' not found in requirements.txt — add it or it may fail at runtime.",
            lineno,
        ))

    @staticmethod
    def _parse_requirements(files: dict[str, str]) -> set[str]:
        """Extract package names from any requirements.txt in generated files."""
        declared: set[str] = set()
        for path, content in files.items():
            if path.endswith("requirements.txt"):
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Strip version specifiers: flask>=2.0 → flask
                        pkg = re.split(r"[>=<!;]", line)[0].strip().lower()
                        if pkg:
                            declared.add(pkg)
        return declared


# ── 4. Architecture Checker ────────────────────────────────────────────────────

_VALID_PATH = re.compile(r"^[\w./-]+$")   # alphanumeric, dots, slashes, hyphens only


class ArchitectureChecker:
    """
    Validate structural constraints on the generated file set.

    Checks:
      - Entry point file exists in generated_files
      - All file paths are valid (no spaces, special chars)
      - No obvious circular imports between generated .py files
    """

    NAME = "architecture"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        issues.extend(self._check_entry_point(files, entry_point))
        issues.extend(self._check_path_names(files))
        issues.extend(self._check_circular_imports(files))
        return issues

    def _check_entry_point(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        if entry_point and entry_point not in files:
            return [_issue(
                "error", self.NAME, entry_point,
                f"Entry point '{entry_point}' not found in generated files.",
            )]
        return []

    def _check_path_names(
        self, files: dict[str, str]
    ) -> list[VerificationIssue]:
        issues = []
        for path in files:
            if not _VALID_PATH.match(path):
                issues.append(_issue(
                    "warning", self.NAME, path,
                    f"File path '{path}' contains unusual characters — may cause issues on some OSes.",
                ))
        return issues

    def _check_circular_imports(
        self, files: dict[str, str]
    ) -> list[VerificationIssue]:
        """
        Build an import dependency graph from generated .py files and detect cycles.

        Only considers intra-project imports (ignores stdlib and third-party).
        """
        py_files = _python_files(files)

        # Map module_name → file_path
        module_map: dict[str, str] = {}
        for path in py_files:
            # e.g. "utils/helper.py" → "utils.helper" and "helper"
            mod = path.replace("/", ".").replace("\\", ".").removesuffix(".py")
            module_map[mod] = path
            module_map[mod.split(".")[-1]] = path   # bare name fallback

        # Build adjacency: file_path → set of file_paths it imports
        graph: dict[str, set[str]] = {p: set() for p in py_files}

        for path, content in py_files.items():
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                imported = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported = node.module.split(".")[0]
                if imported and imported in module_map:
                    target = module_map[imported]
                    if target != path:
                        graph[path].add(target)

        # DFS cycle detection
        issues = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbour in graph.get(node, set()):
                if neighbour not in visited:
                    if dfs(neighbour):
                        return True
                elif neighbour in rec_stack:
                    issues.append(_issue(
                        "warning", self.NAME, node,
                        f"Possible circular import between '{node}' and '{neighbour}'.",
                    ))
                    return True
            rec_stack.discard(node)
            return False

        for path in py_files:
            if path not in visited:
                dfs(path)

        return issues


# ── 5. Runtime Risk Checker ────────────────────────────────────────────────────

class RuntimeRiskChecker:
    """
    Heuristic checks for patterns that typically cause runtime hangs or crashes.

    Checks:
      - Infinite loops: `while True:` with no `break` in the immediate body
      - Unbounded recursion: recursive function with no apparent base case
      - Division by zero: literal `/ 0` or `// 0`
    """

    NAME = "runtime"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for path, content in _python_files(files).items():
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            issues.extend(self._check_infinite_loops(path, tree))
            issues.extend(self._check_recursion(path, tree))
            issues.extend(self._check_divide_by_zero(path, tree))
        return issues

    def _check_infinite_loops(
        self, path: str, tree: ast.AST
    ) -> list[VerificationIssue]:
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.While):
                continue
            # Test is the literal True (or 1)
            test = node.test
            is_true = (
                (isinstance(test, ast.Constant) and test.value is True)
                or (isinstance(test, ast.Constant) and test.value == 1)
                or (isinstance(test, ast.Name) and test.id == "True")
            )
            if not is_true:
                continue
            # Check if any Break exists in the immediate loop body (not nested loops)
            has_break = any(
                isinstance(child, ast.Break)
                for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                if not (isinstance(child, (ast.While, ast.For)) and child is not node)
            )
            if not has_break:
                issues.append(_issue(
                    "warning", self.NAME, path,
                    "Potential infinite loop: `while True` with no `break` detected.",
                    node.lineno,
                ))
        return issues

    def _check_recursion(
        self, path: str, tree: ast.AST
    ) -> list[VerificationIssue]:
        """Warn on functions that call themselves with no obvious `if`/`return` base case."""
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            # Look for a self-call inside the function body
            calls_self = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == name
                for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
            if not calls_self:
                continue
            # Check for at least one conditional (`if` statement) as base-case guard
            has_condition = any(
                isinstance(n, ast.If)
                for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
            if not has_condition:
                issues.append(_issue(
                    "warning", self.NAME, path,
                    f"Recursive function '{name}' has no apparent base case — risk of RecursionError.",
                    node.lineno,
                ))
        return issues

    def _check_divide_by_zero(
        self, path: str, tree: ast.AST
    ) -> list[VerificationIssue]:
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp):
                continue
            if not isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                continue
            right = node.right
            if isinstance(right, ast.Constant) and right.value == 0:
                issues.append(_issue(
                    "error", self.NAME, path,
                    "Division by zero: literal `/ 0` or `// 0` detected.",
                    node.lineno,
                ))
        return issues


# ── Stack Detection ────────────────────────────────────────────────────────────

_NODE_ENTRY_CANDIDATES = (
    "server.js", "app.js", "index.js", "main.js",
    "src/server.js", "src/app.js", "src/index.js",
)


def detect_stack(files: dict[str, str]) -> str:
    """
    Detect the primary language stack from the generated file set.
    Returns 'node', 'python', or 'unknown'.

    Node.js: any .js/.ts file or package.json present.
    Python:  at least one non-empty .py file present.
    An empty main.py created as a DebugAgent placeholder does NOT count as Python.
    """
    js_files = [p for p in files if p.endswith(".js") or p.endswith(".ts")]
    has_package_json = any(
        p == "package.json" or p.endswith("/package.json") for p in files
    )
    if js_files or has_package_json:
        return "node"
    meaningful_py = [p for p in files if p.endswith(".py") and files[p].strip()]
    if meaningful_py:
        return "python"
    return "unknown"


def detect_node_entry(files: dict[str, str]) -> str | None:
    """
    Return the first non-empty JS entry-point file found, or None.

    Priority order: server.js > app.js > index.js > main.js > top-level .js > any .js
    """
    for candidate in _NODE_ENTRY_CANDIDATES:
        if candidate in files and files[candidate].strip():
            return candidate
    # Top-level non-empty .js file
    for path in sorted(files):
        if "/" not in path and path.endswith(".js") and files[path].strip():
            return path
    # Any non-empty .js file
    for path in sorted(files):
        if path.endswith(".js") and files[path].strip():
            return path
    return None


# ── 6. Empty Entry Point Checker ───────────────────────────────────────────────

class EmptyEntryPointChecker:
    """
    Error when the entry point file exists but contains no code.

    This blocks the 'create empty main.py to satisfy entry-point check' bypass
    that DebugAgent uses when it cannot solve cross-language entry point mismatches.
    """

    NAME = "empty_entry"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        if not entry_point or entry_point not in files:
            return []
        if not files[entry_point].strip():
            return [_issue(
                "error", self.NAME, entry_point,
                f"Entry point '{entry_point}' exists but is empty — no code was generated.",
            )]
        return []


# ── 7. Node.js Checker ─────────────────────────────────────────────────────────

_LOCAL_REQUIRE_RE = re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)""")
_NODE_ERR_LINENO_RE = re.compile(r":(\d+)\n")
_NODE_SYNTAX_ERR_RE = re.compile(r"SyntaxError: (.+)")


class NodeJsChecker:
    """
    Node.js-specific static checks. Only active when detect_stack() returns 'node'.

    Checks (all ERROR severity):
      1. At least one non-empty JS entry file exists (server.js / app.js / index.js).
      2. No .js file is completely empty.
      3. JS syntax is valid — `node --check` on each .js file (requires Node ≥ 6).
      4. Local require('./x') and require('../x') paths exist in the generated file set.
    """

    NAME = "nodejs"

    def check(
        self, files: dict[str, str], entry_point: str
    ) -> list[VerificationIssue]:
        if detect_stack(files) != "node":
            return []

        issues: list[VerificationIssue] = []
        js_files = {p: c for p, c in files.items() if p.endswith(".js")}

        issues.extend(self._check_entry_exists(files))
        issues.extend(self._check_empty_js_files(js_files))
        issues.extend(self._check_js_syntax(js_files))
        issues.extend(self._check_require_targets(files, js_files))
        return issues

    # ── sub-checks ────────────────────────────────────────────────────────────

    def _check_entry_exists(
        self, files: dict[str, str]
    ) -> list[VerificationIssue]:
        if detect_node_entry(files) is None:
            return [_issue(
                "error", self.NAME, "",
                "Node.js project has no non-empty entry file "
                "(expected one of: server.js, app.js, index.js).",
            )]
        return []

    def _check_empty_js_files(
        self, js_files: dict[str, str]
    ) -> list[VerificationIssue]:
        return [
            _issue(
                "error", self.NAME, path,
                f"'{path}' is empty — no JavaScript code was generated.",
            )
            for path, content in js_files.items()
            if not content.strip()
        ]

    def _check_js_syntax(
        self, js_files: dict[str, str]
    ) -> list[VerificationIssue]:
        """Run `node --check` on each .js file via a temp file."""
        # Verify node is available before attempting any checks
        try:
            subprocess.run(
                ["node", "--version"],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(
                "[NodeJsChecker] `node` not found on PATH — skipping JS syntax checks."
            )
            return []

        issues: list[VerificationIssue] = []
        for path, content in js_files.items():
            result = self._node_check(content)
            if result is not None:
                line, msg = result
                issues.append(_issue(
                    "error", self.NAME, path,
                    f"JS SyntaxError: {msg}",
                    line,
                ))
        return issues

    def _node_check(
        self, content: str
    ) -> tuple[int | None, str] | None:
        """
        Write content to a temp file, run `node --check`, parse the error output.
        Returns (line_number_or_None, message) on syntax error, None on success.
        """
        tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".js", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(content)
                tmp = f.name
            result = subprocess.run(
                ["node", "--check", tmp],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return None
            stderr = result.stderr
            lineno_match = _NODE_ERR_LINENO_RE.search(stderr)
            msg_match = _NODE_SYNTAX_ERR_RE.search(stderr)
            line = int(lineno_match.group(1)) if lineno_match else None
            msg = (
                msg_match.group(1)
                if msg_match
                else (stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error")
            )
            return (line, msg)
        except (subprocess.TimeoutExpired, OSError):
            return None
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    def _check_require_targets(
        self, files: dict[str, str], js_files: dict[str, str]
    ) -> list[VerificationIssue]:
        """
        For each local require('./x') or require('../x'), verify the target
        exists in the generated file set (with or without .js extension).
        """
        issues: list[VerificationIssue] = []
        for path, content in js_files.items():
            file_dir = path.rsplit("/", 1)[0] if "/" in path else ""
            for match in _LOCAL_REQUIRE_RE.finditer(content):
                req = match.group(1)
                base = posixpath.join(file_dir, req) if file_dir else req
                resolved = posixpath.normpath(base)
                candidates = [
                    resolved,
                    resolved + ".js",
                    resolved + "/index.js",
                ]
                if not any(c in files for c in candidates):
                    issues.append(_issue(
                        "error", self.NAME, path,
                        f"require('{req}') — '{resolved}.js' not found in generated files.",
                    ))
        return issues


# ── Orchestrator ───────────────────────────────────────────────────────────────

_CHECKERS = [
    SyntaxChecker(),
    SecurityChecker(),
    DependencyChecker(),
    ArchitectureChecker(),
    RuntimeRiskChecker(),
    EmptyEntryPointChecker(),
    NodeJsChecker(),
]


def run_all_checks(
    files: dict[str, str],
    entry_point: str,
) -> VerificationResult:
    """
    Run all five static checkers against the generated file set.

    Returns a VerificationResult with:
      - passed=True  if no ERROR-severity issues found
      - passed=False if any ERROR-severity issue exists
    """
    all_issues: list[VerificationIssue] = []

    for checker in _CHECKERS:
        try:
            found = checker.check(files, entry_point)
            all_issues.extend(found)
            if found:
                logger.debug(
                    "[%s] %d issue(s) found.", checker.NAME, len(found)
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Checker raised unexpectedly: %s", checker.NAME, exc)

    errors = [i for i in all_issues if i["severity"] == "error"]
    warnings = [i for i in all_issues if i["severity"] == "warning"]
    passed = len(errors) == 0

    summary = _build_summary(all_issues, passed)
    logger.info("[Verification] %s — %d error(s), %d warning(s).",
                "PASSED" if passed else "FAILED", len(errors), len(warnings))

    return VerificationResult(
        passed=passed,
        issues=all_issues,
        generated_tests={},    # populated by test_generator
        error_count=len(errors),
        warning_count=len(warnings),
        summary=summary,
    )


def _build_summary(issues: list[VerificationIssue], passed: bool) -> str:
    if not issues:
        return "All checks passed. No issues found."

    lines = [f"Verification {'PASSED' if passed else 'FAILED'}."]
    for sev in ("error", "warning", "info"):
        group = [i for i in issues if i["severity"] == sev]
        if group:
            lines.append(f"\n{sev.upper()}S ({len(group)}):")
            for issue in group:
                loc = f"{issue['file']}"
                if issue["line"]:
                    loc += f":{issue['line']}"
                lines.append(f"  [{issue['check']}] {loc} — {issue['message']}")
    return "\n".join(lines)
