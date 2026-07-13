"""Deterministic solvers reused across categories.

Each returns a string answer or None when it cannot confidently solve the
task. The router only uses these when they return a value, so returning None
is the correct "I don't know" signal and lets the local model attempt instead.

All execution surfaces here are sandboxed: math programs run under an AST
whitelist with a restricted builtin namespace; bug-fix templates rewrite the
source tree and re-parse rather than exec untrusted code.
"""
from __future__ import annotations

import ast
import itertools
import math
import re
from typing import Any, Optional

__all__ = ["program_aided_math", "solve_logic", "fix_code_bug"]


# --------------------------------------------------------------------------
# Program-aided math
# --------------------------------------------------------------------------
# A word problem is rewritten into a tiny straight-line Python program and
# executed in a sandbox. This covers multi-step arithmetic the single-expression
# evaluator in router.py cannot (e.g. "has 3, gives 2, buys 5").
_SAFE_CALLS = {
    "abs": abs, "len": len, "max": max, "min": min,
    "round": round, "sum": sum,
}
_ALLOWED_NODES = (
    ast.Module, ast.Assign, ast.Expr, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Call,
    ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.Compare, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And,
    ast.Or, ast.IfExp, ast.keyword,
)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)


def _math_source(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    block = _CODE_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    text = re.sub(r"^python\s*\n", "", text, flags=re.I)
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"[$€£,](?=\d)", "", text)
    if re.fullmatch(r"[\d\s+\-*/().%]+\s*=\s*-?\d+(?:\.\d+)?", text):
        text = text.rsplit("=", 1)[0].strip()
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^(?:answer|program|calculation)\s*:\s*", s, re.I):
            s = s.split(":", 1)[1].strip()
        lines.append(s)
    out = "\n".join(lines)
    if out and "\n" not in out and "=" not in out and re.fullmatch(r"[\d\s+\-*/().%]+", out) and re.search(r"[+\-*/%]", out):
        return f"result = {out}"
    return out or None


def _safe_tree(tree: ast.Module) -> bool:
    if not 1 <= len(tree.body) <= 32:
        return False
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return False
            if not math.isfinite(float(node.value)) or abs(float(node.value)) > 1e15:
                return False
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return False
            if node.targets[0].id.startswith("_") or not node.targets[0].id.isidentifier():
                return False
            assigned.add(node.targets[0].id)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in assigned and node.id not in _SAFE_CALLS:
                return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS:
                return False
            if len(node.args) > 20 or node.keywords:
                return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or abs(float(node.right.value)) > 12:
                return False
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 100:
            return False
    return "result" in assigned


def _format_result(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value - round(value)) < 1e-10:
            return str(round(value))
        return f"{value:.10f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)) and 1 <= len(value) <= 8:
        parts = [_format_result(i) for i in value]
        if all(p is not None for p in parts):
            return ", ".join(parts)
    return None


def program_aided_math(prompt: str) -> Optional[str]:
    """Run a model-produced math program in a sandbox; return its result."""
    # We do not generate the program ourselves; we only execute one that the
    # caller has placed in a code fence. The local model step produces it, then
    # this validates and runs it. If no fence is present we cannot proceed.
    src = _math_source(prompt)
    if not src or len(src) > 1200:
        return None
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError:
        return None
    if not _safe_tree(tree):
        return None
    ns: dict[str, Any] = {}
    try:
        exec(compile(tree, "<math>", "exec"), {"__builtins__": _SAFE_CALLS}, ns)
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return _format_result(ns.get("result"))


# --------------------------------------------------------------------------
# Logic solver: comparison chains + assignment puzzles + syllogisms
# --------------------------------------------------------------------------
_COMPARATIVES = {
    "taller": "shorter", "older": "younger", "faster": "slower", "bigger": "smaller",
    "heavier": "lighter", "stronger": "weaker", "richer": "poorer", "higher": "lower",
    "earlier": "later",
}
_INVERSE = {low: high for high, low in _COMPARATIVES.items()}
_SUPERLATIVE = {h: f"{h}est" for h in _COMPARATIVES}
_ENTITY = r"[A-Z][a-zA-Z]*|(?:[Aa]n?|[Tt]he)\s+[a-z]+"
_COMPARISON_RE = re.compile(
    rf"\b({_ENTITY})\s+(?:is|was|runs?|ran)\s+({ '|'.join(list(_COMPARATIVES)+list(_INVERSE)) })\s+than\s+({_ENTITY})\b"
)
_ASSIGN_VERBS = r"owns?|has|have|likes?|drinks?|plays?|drives?|wears?|reads?|prefers?|keeps?|stud(?:y|ies|ied)"
_NAME = r"[A-Z][a-z]+"
_POS_FACT_RE = re.compile(rf"\b({_NAME})\s+(?:{_ASSIGN_VERBS})\s+(?:a\s+|an\s+|the\s+)?(\w+)", re.I)
_NEG_FACT_RE = re.compile(
    rf"\b({_NAME})\s+(?:does\s+not|doesn'?t|do\s+not|don'?t|did\s+not|didn'?t|cannot|can'?t|will\s+not|won'?t|never)\s+"
    rf"(?:{_ASSIGN_VERBS})\s+(?:a\s+|an\s+|the\s+)?(\w+)", re.I)
_WHO_RE = re.compile(rf"\bwho\s+(?:{_ASSIGN_VERBS})\s+(?:a\s+|an\s+|the\s+)?(\w+)", re.I)
_WHAT_RE = re.compile(rf"\bwhat\s+(?:does\s+)?({_NAME})\s+(?:{_ASSIGN_VERBS})", re.I)
_SYLLOGISM_RE = re.compile(
    r"\b(?:if\s+)?all\s+([A-Z][A-Za-z]*)\s+are\s+([A-Z][A-Za-z]*)\s+and\s+all\s+(\2)\s+are\s+([A-Z][A-Za-z]*)\b", re.I)
_ITEMS_RE = re.compile(
    rf"(?:{_ASSIGN_VERBS})\s+(?:a\s+|an\s+|the\s+)?different\s+\w+\s*[:\-]\s*"
    rf"([\w\s,]+?)(?:\.|$)", re.I)
_STOPWORDS = {"who", "what", "which", "the", "a", "an", "each", "every", "different",
              "and", "or", "three", "two", "four", "five", "friends", "people", "person"}


def _norm(raw: str) -> str:
    return re.sub(r"^(?:an?|the)\s+", "", raw.strip(), flags=re.I)


def _topo(edges):
    nodes, graph, indeg = set(), {}, {}
    for hi, lo in edges:
        nodes.update((hi, lo))
        graph.setdefault(hi, set()).add(lo)
        indeg[lo] = indeg.get(lo, 0) + 1
        indeg.setdefault(hi, 0)
    q = sorted(n for n in nodes if indeg.get(n, 0) == 0)
    order = []
    while q:
        if len(q) > 1:
            return None
        n = q.pop(0)
        order.append(n)
        for nxt in sorted(graph.get(n, ())):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return order if len(order) == len(nodes) else None


def _extreme(edges, first):
    nodes, graph, indeg = set(), {}, {}
    for hi, lo in edges:
        nodes.update((hi, lo))
        graph.setdefault(hi, set()).add(lo)
        indeg.setdefault(hi, 0)
        indeg.setdefault(lo, 0)
        indeg[lo] += 1
    q = [n for n in nodes if indeg.get(n, 0) == 0]
    seen = 0
    while q:
        seen += 1
        n = q.pop(0)
        for nxt in graph.get(n, ()):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if seen != len(nodes):
        return None
    cands = [n for n in nodes if indeg.get(n, 0) == 0] if first else [n for n in nodes if not graph.get(n)]
    return cands[0] if len(cands) == 1 else None


def _solve_assignment(prompt: str):
    q_item = _WHO_RE.search(prompt)
    q_name = _WHAT_RE.search(prompt)
    if not q_item and not q_name:
        return None
    names = re.findall(rf"\b{_NAME}\b", prompt)
    names = list(dict.fromkeys(n for n in names if n.lower() not in _STOPWORDS))[:5]
    m = _ITEMS_RE.search(prompt)
    items = []
    if m:
        raw = re.sub(r"\b(a|an|the|and|or)\b", ",", m.group(1), flags=re.I)
        items = [i.strip().lower() for i in raw.split(",") if i.strip()]
        items = [i for i in items if i and i not in _STOPWORDS][:5]
    if not (2 <= len(names) == len(items) <= 5):
        return None
    low = {i.lower(): i for i in items}
    pos, neg = [], []
    for nm, it in _POS_FACT_RE.findall(prompt):
        if nm in names and it.lower() in low:
            pos.append((nm, it.lower()))
    for nm, it in _NEG_FACT_RE.findall(prompt):
        if nm in names and it.lower() in low:
            neg.append((nm, it.lower()))
            pos = [f for f in pos if f != (nm, it.lower())]
    if not pos and not neg:
        return None
    sols = []
    for perm in itertools.permutations(sorted(low)):
        a = dict(zip(names, perm))
        if all(a[n] == i for n, i in pos) and all(a[n] != i for n, i in neg):
            sols.append(a)
    if not sols:
        return None
    if q_item:
        want = q_item.group(1).lower()
        if want in low:
            owners = {n for sol in sols for n, i in sol.items() if i == want}
            if len(owners) == 1:
                return f"{owners.pop()} {_verb(q_item.group(0))} the {want}."
    if q_name and q_name.group(1) in names:
        owned = {sol[q_name.group(1)] for sol in sols}
        if len(owned) == 1:
            return f"{q_name.group(1)} {_verb(q_name.group(0))} the {owned.pop()}."
    return None


def _verb(text: str) -> str:
    v = re.search(rf"\b({_ASSIGN_VERBS})\b", text, re.I)
    v = v.group(1).lower() if v else "owns"
    if v in {"has", "studies"}:
        return v
    if v == "have":
        return "has"
    return v if v.endswith("s") else f"{v}s"


def _solve_comparisons(prompt: str):
    edges = []
    rel = None
    for left, r, right in _COMPARISON_RE.findall(prompt):
        left, right = _norm(left), _norm(right)
        base = _INVERSE.get(r, r)
        if rel is None:
            rel = base
        elif rel != base:
            return None
        hi, lo = (right, left) if r in _INVERSE else (left, right)
        edges.append((hi, lo))
    if not edges or rel is None:
        return None
    q = prompt.rfind("?")
    clause = prompt[max(prompt.rfind(".", 0, q), prompt.rfind("!", 0, q)) + 1:q + 1].strip().lower()
    sup = _SUPERLATIVE.get(rel, rel)
    inv = _SUPERLATIVE.get(_COMPARATIVES.get(rel, ""), "")
    asks_hi = bool(re.search(rf"\b(?:the\s+)?{sup}\b", clause))
    asks_lo = bool(inv) and re.search(rf"\b(?:the\s+)?{inv}\b", clause)
    if asks_hi == asks_lo:
        order = _topo(edges)
        return " > ".join(order) if order else None
    ans = _extreme(edges, asks_hi)
    if not ans:
        return None
    adj = sup if asks_hi else inv
    return f"{ans if ans[0].isupper() else 'The ' + ans} is the {adj}."


def solve_logic(prompt: str) -> Optional[str]:
    syl = _SYLLOGISM_RE.search(prompt)
    if syl and re.search(r"\b(must|necessarily|always|every)\b", prompt, re.I):
        a, _, c = syl.groups()
        if re.search(rf"\bevery\s+{re.escape(a.rstrip('s'))}s?\s+(?:be|is|are)\s+(?:an?\s+)?{re.escape(c.rstrip('s'))}", prompt, re.I):
            return f"Yes. Every {a.rstrip('s')} is a {c.rstrip('s')}."
    a = _solve_assignment(prompt)
    if a:
        return a
    return _solve_comparisons(prompt)


# --------------------------------------------------------------------------
# AST bug-fixer (canonical, single-function shapes)
# --------------------------------------------------------------------------
def _single_func(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    if tree.body[0].decorator_list:
        return None
    return tree, tree.body[0]


def _render(tree):
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _simple_params(fn, n):
    a = fn.args
    if a.posonlyargs or len(a.args) != n or a.vararg or a.kwonlyargs or a.kwarg or a.defaults:
        return None
    return [x.arg for x in a.args]


def fix_code_bug(code: str, instruction: str) -> Optional[str]:
    """Return corrected source for a small set of well-known bug shapes."""
    parsed = _single_func(code)
    if not parsed:
        return None
    tree, fn = parsed
    ins = instruction.lower()
    p = _simple_params(fn, 1)
    if not p:
        return None
    param = p[0]

    # early return inside accumulation loop
    if len(fn.body) == 2 and isinstance(fn.body[0], ast.Assign) and _const(fn.body[0].value, 0) \
       and isinstance(fn.body[1], ast.For) and isinstance(fn.body[1].body[-1], ast.Return):
        loop = fn.body[1]
        if isinstance(loop.body[0], ast.AugAssign) and isinstance(loop.body[0].op, ast.Add):
            loop.body.pop()
            fn.body.append(loop.body.pop())
            return _render(tree)

    # factorial base 0 -> 1
    if re.fullmatch(r"(?:[A-Za-z_]\w*_)?factorial", fn.name, re.I) and len(fn.body) == 2 \
       and isinstance(fn.body[0], ast.If) and _const(fn.body[0].body[0].value, 0):
        fn.body[0].body[0].value = ast.Constant(1)
        return _render(tree)

    # return first element instead of max/min
    if len(fn.body) == 1 and isinstance(fn.body[0], ast.Return) and isinstance(fn.body[0].value, ast.Subscript):
        nm = fn.name.lower()
        op = "max" if ("max" in nm or re.search(r"\bmax(?:imum)?\b", ins)) else "min" if ("min" in nm or re.search(r"\bmin(?:imum)?\b", ins)) else None
        if op:
            fn.body[0].value = ast.Call(func=ast.Name(op, ast.Load()), args=[ast.Name(param, ast.Load())], keywords=[])
            return _render(tree)

    # off-by-one: range(len(x)-1) when indexing every element
    if re.search(r"\b(off[- ]by[- ]one|miss(?:es|ing)? (?:the )?last|every element|all elements)\b", ins):
        for node in ast.walk(fn):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and _name(node.iter.func, "range") \
               and isinstance(node.iter.args[0], ast.BinOp) and isinstance(node.iter.args[0].op, ast.Sub) \
               and _const(node.iter.args[0].right, 1):
                node.iter.args[0] = node.iter.args[0].left
                return _render(tree)
    return None


def _const(node, val):
    return isinstance(node, ast.Constant) and node.value == val


def _name(node, id_):
    return isinstance(node, ast.Name) and node.id == id_
