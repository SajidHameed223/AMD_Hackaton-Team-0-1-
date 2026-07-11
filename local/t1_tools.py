"""Bounded, allow-listed tools for the Track 1 local inference harness."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}
_CALC_FUNCS = {"abs": abs, "ceil": math.ceil, "floor": math.floor, "round": round, "sqrt": math.sqrt}
_CALC_CONSTS = {"e": math.e, "pi": math.pi}
_SAFE_IMPORTS = {"collections", "functools", "itertools", "json", "math", "re", "statistics", "string"}
_DENIED_NAMES = {"__builtins__", "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec", "getattr", "globals", "hasattr", "help", "input", "locals", "memoryview", "object", "open", "os", "setattr", "socket", "subprocess", "super", "sys", "type", "urllib", "vars"}
_RUNNER = r'''
import builtins
import pathlib
import sys

ALLOWED = {"collections", "functools", "itertools", "json", "math", "re", "statistics", "string"}
SAFE = {name: getattr(builtins, name) for name in (
    "AssertionError", "Exception", "TypeError", "ValueError",
    "__build_class__", "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "int", "isinstance", "len", "list", "map", "max", "min", "next", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
)}
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".", 1)[0] not in ALLOWED:
        raise ImportError("import is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)
SAFE["__import__"] = guarded_import
path = sys.argv[1]
source = pathlib.Path(path).read_text(encoding="utf-8")
namespace = {"__name__": "__main__", "__builtins__": SAFE}
exec(compile(source, path, "exec"), namespace, namespace)
'''


def safe_calculate(expression: str) -> int | float:
    """Evaluate a numeric expression without exposing Python objects or eval."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("calculator expression must be non-empty")
    if len(expression) > 240:
        raise ValueError("calculator expression exceeds 240 characters")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def visit(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 20:
            raise ValueError("calculator expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CALC_CONSTS:
            return _CALC_CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("calculator exponent is out of bounds")
            result = _BINARY[type(node.op)](left, right)
            if not math.isfinite(float(result)) or abs(float(result)) > 1e100:
                raise ValueError("calculator result is out of bounds")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand, depth + 1))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCS and not node.keywords and len(node.args) <= 2:
            return _CALC_FUNCS[node.func.id](*(visit(arg, depth + 1) for arg in node.args))
        raise ValueError("calculator expression contains an unsupported operation")

    result = visit(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return round(result, 12) if isinstance(result, float) else result


def check_python_syntax(source: str) -> dict[str, Any]:
    if not isinstance(source, str) or not source.strip():
        return {"valid": False, "error": "No Python source supplied"}
    if len(source) > 12_000:
        return {"valid": False, "error": "Python source exceeds 12000 characters"}
    try:
        ast.parse(source)
        return {"valid": True, "error": None}
    except SyntaxError as exc:
        return {"valid": False, "error": f"line {exc.lineno}: {exc.msg}"}


def _validate_executable_python(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DENIED_NAMES:
            raise ValueError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in _SAFE_IMPORTS:
                    raise ValueError(f"forbidden import: {alias.name}")
        if isinstance(node, ast.ImportFrom) and (not node.module or node.module.split(".", 1)[0] not in _SAFE_IMPORTS):
            raise ValueError(f"forbidden import: {node.module or 'relative import'}")


def execute_python(source: str) -> dict[str, Any]:
    """Run safe algorithmic Python in a disposable, resource-limited process."""

    syntax = check_python_syntax(source)
    if not syntax["valid"]:
        return {"ok": False, "error": syntax["error"], "stdout": "", "stderr": ""}
    try:
        _validate_executable_python(source)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}

    timeout = max(1.0, min(float(os.getenv("LOCAL_PYTHON_TIMEOUT_S", "2")), 8.0))
    memory_mb = max(64, min(int(os.getenv("LOCAL_PYTHON_MEMORY_MB", "128")), 512))
    output_bytes = max(2_048, min(int(os.getenv("LOCAL_PYTHON_OUTPUT_BYTES", "8_192")), 65_536))

    def limits() -> None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout)), max(1, int(timeout) + 1)))
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except Exception:
            pass

    with tempfile.TemporaryDirectory(prefix="o1-t1-") as temp_dir:
        program = Path(temp_dir) / "candidate.py"
        program.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER, str(program)],
                cwd=temp_dir,
                env={"PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=limits if os.name != "nt" else None,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"execution exceeded {timeout}s", "stdout": "", "stderr": ""}
    stdout, stderr = completed.stdout[:output_bytes], completed.stderr[:output_bytes]
    return {"ok": completed.returncode == 0, "error": None if completed.returncode == 0 else f"exit code {completed.returncode}", "stdout": stdout, "stderr": stderr}


def _topic_results(topics: list[Any]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        if isinstance(topic.get("Topics"), list):
            results.extend(_topic_results(topic["Topics"]))
        elif topic.get("Text"):
            results.append({"title": str(topic["Text"])[:160], "snippet": str(topic["Text"])[:700], "url": str(topic.get("FirstURL", ""))[:500]})
    return results


def web_search(query: str) -> dict[str, Any]:
    """Get bounded JSON search evidence; disabled unless explicitly configured."""

    query = str(query).strip()[:300]
    if os.getenv("LOCAL_WEB_SEARCH_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        return {"available": False, "query": query, "results": [], "error": "web search disabled"}
    if not query:
        return {"available": False, "query": query, "results": [], "error": "empty query"}
    endpoint = os.getenv("LOCAL_WEB_SEARCH_ENDPOINT", "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1")
    separator = "&" if "?" in endpoint else "?"
    url = f"{endpoint}{separator}{urllib.parse.urlencode({'q': query})}"
    headers = {"Accept": "application/json", "User-Agent": "Team-O1-Track1/1.0"}
    if os.getenv("LOCAL_WEB_SEARCH_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LOCAL_WEB_SEARCH_API_KEY']}"
    timeout = max(1.0, min(float(os.getenv("LOCAL_WEB_SEARCH_TIMEOUT_S", "4")), 10.0))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            payload = json.loads(response.read(256_000).decode("utf-8"))
    except Exception as exc:
        return {"available": False, "query": query, "results": [], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    results: list[dict[str, str]] = []
    if isinstance(payload, dict) and payload.get("AbstractText"):
        results.append({"title": str(payload.get("Heading", "Answer"))[:160], "snippet": str(payload["AbstractText"])[:700], "url": str(payload.get("AbstractURL", ""))[:500]})
    if isinstance(payload, dict) and isinstance(payload.get("RelatedTopics"), list):
        results.extend(_topic_results(payload["RelatedTopics"]))
    if isinstance(payload, dict) and isinstance(payload.get("results") or payload.get("items"), list):
        for item in payload.get("results") or payload.get("items"):
            if isinstance(item, dict):
                results.append({"title": str(item.get("title", item.get("name", "Result")))[:160], "snippet": str(item.get("snippet", item.get("description", item.get("text", ""))))[:700], "url": str(item.get("url", item.get("link", "")))[:500]})
    limit = max(1, min(int(os.getenv("LOCAL_WEB_SEARCH_RESULTS", "3")), 5))
    return {"available": bool(results), "query": query, "results": results[:limit], "error": None}


def execute_tool(request: dict[str, Any]) -> dict[str, Any]:
    name = str(request.get("name", "")).strip().lower()
    tool_input = str(request.get("input", request.get("query", "")))
    try:
        if name == "calculator":
            return {"tool": name, "ok": True, "input": tool_input[:240], "result": safe_calculate(tool_input)}
        if name == "web_search":
            result = web_search(tool_input)
            return {"tool": name, "ok": result["available"], **result}
        if name == "python_syntax":
            result = check_python_syntax(tool_input)
            return {"tool": name, "ok": result["valid"], "result": result}
        if name == "python_execute":
            result = execute_python(tool_input)
            return {"tool": name, "ok": result["ok"], "result": result}
        if name == "current_time":
            return {"tool": name, "ok": True, "result": datetime.now(timezone.utc).isoformat()}
        return {"tool": name or "unknown", "ok": False, "error": "tool is not allow-listed"}
    except Exception as exc:
        return {"tool": name or "unknown", "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}


def execute_requests(requests: Any) -> list[dict[str, Any]]:
    """Run at most three unique analyzer-requested tools."""
    if not isinstance(requests, list):
        return []
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for request in requests[:3]:
        if not isinstance(request, dict):
            continue
        key = (str(request.get("name", "")).strip().lower(), str(request.get("input", request.get("query", ""))).strip()[:300])
        if key not in seen:
            seen.add(key)
            evidence.append(execute_tool(request))
    return evidence
