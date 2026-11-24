#!/usr/bin/env python3
"""Guards for the Google-style documentation pass.

Two checks:

* ``same-code``: the module's code is unchanged except for docstrings. Both versions are
  parsed, every docstring is removed, and the AST dumps must be identical. Comments are
  invisible to ``ast``, so editing comments also passes. Use this after a docs-only edit.
* ``coverage``: every function, method and class has a docstring. Every public one that
  is not trivial also has an ``Examples:`` section, plus ``Args:`` when it takes
  parameters other than ``self``/``cls``. In ``tests/test_*.py`` and ``conftest.py`` only
  the docstring itself is required: it must say what property the test checks and why.

Examples:
    Check that a docs-only edit of the normalizer left the code untouched::

        $ python tools/check_docstrings.py same-code utils/normalizer.py
        utils/normalizer.py: code unchanged (docstrings only)

    List functions that still lack the required sections::

        $ python tools/check_docstrings.py coverage utils/normalizer.py
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

_DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def strip_docstrings(source: str) -> str:
    """Return the AST dump of ``source`` with every docstring removed.

    Args:
        source: Python source code.

    Returns:
        ``ast.dump`` of the module with the leading string-constant statement removed
        from the module and from every class and function body.

    Examples:
        >>> a = 'def f(x):\\n    \"\"\"Old.\"\"\"\\n    return x\\n'
        >>> b = 'def f(x):\\n    \"\"\"New, longer.\\n\\n    Examples:\\n        ...\\n    \"\"\"\\n    return x\\n'
        >>> strip_docstrings(a) == strip_docstrings(b)
        True
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _DOC_NODES) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _git_show(ref: str, path: str) -> str:
    """Text of ``path`` at the git revision ``ref``."""
    return subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True, text=True).stdout


def same_code(paths: list[str], ref: str) -> int:
    """Compare each file with its version at ``ref``, ignoring docstrings; print the result per file.

    Args:
        paths: Files to check (paths as git knows them).
        ref: Git revision to compare with.

    Returns:
        0 if every file's code is unchanged, else 1.

    Examples:
        >>> same_code(["tools/check_docstrings.py"], "HEAD")  # doctest: +SKIP
        tools/check_docstrings.py: code unchanged (docstrings only)
        0
    """
    bad = 0
    for path in paths:
        old = strip_docstrings(_git_show(ref, path))
        new = strip_docstrings(Path(path).read_text())
        if old == new:
            print(f"{path}: code unchanged (docstrings only)")
        else:
            print(f"{path}: CODE CHANGED relative to {ref}")
            bad += 1
    return 1 if bad else 0


def _is_trivial(node: ast.AST) -> bool:
    """True if the function body, docstring excluded, is at most one statement (no ``Examples:`` required)."""
    body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    return len(body) <= 1


def coverage(paths: list[str]) -> int:
    """Report functions/classes missing a docstring, ``Args:`` or ``Examples:``; return a status code."""
    missing = 0
    for path in paths:
        tree = ast.parse(Path(path).read_text())
        name = Path(path).name
        is_test_file = name.startswith("test_") or name == "conftest.py"
        # Functions nested inside other functions are local helpers: they need a docstring
        # but not the public-API sections, so mark them before walking.
        nested = set()
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(parent):
                    if child is not parent and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        nested.add(id(child))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            where = f"{path}:{node.lineno} {node.name}"
            if not doc:
                print(f"{where}: no docstring")
                missing += 1
                continue
            public = not node.name.startswith("_") or node.name == "__init__"
            # Tests only need a docstring saying what they check: their arguments are pytest
            # fixtures and the test body is already the usage example.
            if is_test_file or isinstance(node, ast.ClassDef) or not public or id(node) in nested:
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
            if args and "Args:" not in doc:
                print(f"{where}: missing Args:")
                missing += 1
            if node.name != "__init__" and not _is_trivial(node) and "Examples:" not in doc:
                print(f"{where}: missing Examples:")
                missing += 1
    print(f"{missing} issue(s)")
    return 1 if missing else 0


def main() -> int:
    """Command-line entry point: ``same-code <files> [--ref REF]`` or ``coverage <files>``.

    Returns:
        The process exit status.

    Examples:
        Check the docs-only rule and the coverage of a module::

            python tools/check_docstrings.py same-code utils/analysis.py
            python tools/check_docstrings.py coverage utils/analysis.py
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_same = sub.add_parser("same-code", help="code unchanged except docstrings")
    p_same.add_argument("paths", nargs="+")
    p_same.add_argument("--ref", default="HEAD", help="git ref to compare against (default: HEAD)")
    p_cov = sub.add_parser("coverage", help="docstring / Args / Examples coverage")
    p_cov.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.cmd == "same-code":
        return same_code(args.paths, args.ref)
    return coverage(args.paths)


if __name__ == "__main__":
    sys.exit(main())
