"""Category-classifier + deterministic-solver dispatch (0 Fireworks tokens)."""
from __future__ import annotations

import ast
import builtins
import re
import symtable
from collections import Counter

try:
    from app.track1_router import classify_domain as _track1_classify_domain
    from app.track1_router import deterministic_answer as _track1_deterministic_answer
except Exception:  # pragma: no cover - keeps standalone demo mode working
    _track1_classify_domain = None
    _track1_deterministic_answer = None


_MATH_PAT = re.compile(
    r"\b(calcul|comput|solv|how many|how much|remain|total|sum|differ|product"
    r"|percent|fraction|\d+\s*[\+\-\*\/\%\^]\s*\d+|average|mean|median"
    r"|ratio|proportion|what is \d|equals|result of)\b", re.I)
_SENTIMENT_PAT = re.compile(
    r"\b(sentiment|tone|feeling|emotion|positive|negative|neutral"
    r"|attitude|mood|opinion|optimistic|pessimistic|analyze the (?:sentiment|tone|feeling)"
    r"|how does .* feel|what (?:is|are) the (?:sentiment|tone))\b", re.I)
_NER_PAT = re.compile(
    r"\b(named entit|entities|extract (?:names?|persons?|locations?|organizations?"
    r"|dates?|places?)|identify (?:the )?(?:names?|persons?|people|locations?"
    r"|organizations?|dates?|places?)|who |where .*(?:located|based|born)"
    r"|list (?:all )?(?:the )?(?:names?|persons?|organizations?|locations?))\b", re.I)
_SUMMARIZE_PAT = re.compile(
    r"(?:\bsummar\w*\b|\b(?:brief|overview|condense|shorten|tldr|tl;dr|main points?"
    r"|key (?:points?|takeaways?|ideas?)|gist|recap|in short)\b)", re.I)
_CODE_DEBUG_PAT = re.compile(
    r"\b(debug|fix (?:the |this )?(?:code|bug|error|issue)|syntax error"
    r"|trace ?back|error message|what(?:'s| is) wrong with (?:this|the) code"
    r"|find (?:the )?(?:bug|error|issue)|correct (?:the |this )?code"
    r"|why (?:does|is) (?:this|the) code)\b", re.I)
_CODE_GEN_PAT = re.compile(
    r"\b(write (?:a |the )?(?:\w+ )?(?:function|program|script|code|class|method|module)"
    r"|implement|generate (?:code|a function|a program)|create (?:a )?(?:function"
    r"|program|script|class)|code (?:to|that|for|which)|algorithm (?:to|for|that))\b", re.I)
_LOGICAL_PAT = re.compile(
    r"\b(logic|puzzle|riddle|deduc|infer|syllogism|truth table|premise"
    r"|conclusion|if .* then|constraint|who (?:lives|sits|owns|drinks|eats|painted|drove|built|chose|picked)"
    r"|arrangement|seat|order.*(?:left|right|next)"
    r"|sequence|what comes next|next (?:number|term|in)|pattern"
    r"|each (?:own|painted|drove|built|chose|picked) a different)\b", re.I)
_FACTUAL_PAT = re.compile(
    r"\b(capital of|president of|who (?:is|was|invented|discovered|founded|wrote)"
    r"|when (?:was|did|is)|where (?:is|was)|what (?:is|was|are) the"
    r"|(?:tallest|largest|smallest|longest|deepest|highest|oldest)"
    r"|population of|currency of|founded in|invented by)\b", re.I)

_CLASSIFY_ORDER = [
    (_CODE_DEBUG_PAT, "code_debug"), (_CODE_GEN_PAT, "code_gen"),
    (_SUMMARIZE_PAT, "summarization"), (_SENTIMENT_PAT, "sentiment"),
    (_LOGICAL_PAT, "logical"), (_NER_PAT, "ner"),
    (_MATH_PAT, "math"), (_FACTUAL_PAT, "factual"),
]

def classify(prompt: str) -> str:
    for pat, cat in _CLASSIFY_ORDER:
        if pat.search(prompt):
            return cat
    return "factual"


_SAFE_AST_TYPES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)

def _safe_eval_expr(expr: str) -> float | None:
    expr = expr.strip()
    if not expr or not re.fullmatch(r"[\d\+\-\*\/\(\)\.\s\%]+", expr):
        return None
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, _SAFE_AST_TYPES):
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                continue
            return None
        return eval(compile(tree, "<expr>", "eval"))  # noqa: S307 — AST-validated
    except Exception:
        return None


def solve_math(prompt: str) -> tuple[str, float]:
    direct = re.search(r"(?:what is|calculate|compute|result of)\s+([\d\+\-\*\/\(\)\.\s\%]+)", prompt, re.I)
    if direct:
        val = _safe_eval_expr(direct.group(1))
        if val is not None:
            return (str(int(val)) if val == int(val) else f"{val:.2f}"), 1.0

    pct = re.search(r"(?:has|have|start\w* with|with)\s+(\d+(?:\.\d+)?)\s*(?:items?|products?|units?|things?|people|students?)", prompt, re.I)
    if pct:
        total = float(pct.group(1))
        remaining = total
        for m in re.finditer(r"(?:sells?|removes?|loses?|gives? away|discards?)\s+(\d+(?:\.\d+)?)\s*%", prompt, re.I):
            remaining -= total * float(m.group(1)) / 100
        for m in re.finditer(r"(?:sells?|removes?|loses?|gives? away|discards?)\s+(\d+(?:\.\d+)?)\s+more\b", prompt, re.I):
            remaining -= float(m.group(1))
        for m in re.finditer(r"\band\s+(\d+(?:\.\d+)?)\s+(?:more|additional|extra)\b", prompt, re.I):
            remaining -= float(m.group(1))
        if remaining != total:
            val = remaining
            return (str(int(val)) if val == int(val) else f"{val:.2f}"), 1.0

    # handles "if a class/group has X and Y% are absent, how many are present?" pattern
    # ceiling: chained percentages (X% absent, then Z more left) — defer to T1.
    if re.search(r"\b(?:present|remaining|left|how many)\b", prompt.lower()):
        m_total = re.search(r"\b(?:has|have|with|starts? with|class|group|total of)\s+(\d+(?:\.\d+)?)\s+(?:students?|people|workers?|items?|units?|tickets?|seats?)\b", prompt, re.I)
        m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%\s+(?:are|is)?\s*(?:absent|missing|gone|used|spent|sold|broken|defective|rejected)", prompt, re.I)
        if m_total and m_pct:
            total = float(m_total.group(1))
            absent = total * float(m_pct.group(1)) / 100
            val = total - absent
            return (str(int(val)) if val == int(val) else f"{val:.2f}"), 1.0

    arith = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)", prompt)
    if arith:
        val = _safe_eval_expr(arith.group(1))
        if val is not None:
            return (str(int(val)) if val == int(val) else f"{val:.2f}"), 1.0

    return "", 0.0


_POS_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "brilliant", "outstanding", "superb", "love", "loved", "enjoy", "enjoyed",
    "happy", "pleased", "delighted", "impressive", "beautiful", "best",
    "perfect", "exciting", "incredible", "positive", "awesome", "nice",
    "cheerful", "joyful", "grateful", "optimistic", "thrilled", "enthusiastic",
    "inspired", "uplifting", "charming", "elegant", "magnificent", "splendid",
    "heartwarming", "helpful", "hilarious", "lively", "marvelous",
    "delicious", "tasty", "flavorful", "yummy", "scrumptious", "savory", "fresh",
}
_NEG_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "disappointing",
    "dreadful", "miserable", "ugly", "hate", "hated", "dislike", "angry",
    "sad", "upset", "frustrated", "annoyed", "worst", "boring", "mediocre",
    "disgusting", "nasty", "pathetic", "useless", "negative", "fail",
    "failed", "failure", "broken", "painful", "unhappy", "depressed",
    "pessimistic", "tragic", "devastating", "heartbreaking", "cruel",
    "hostile", "offensive", "toxic", "violent", "appalling", "gloomy",
    "scratch", "scratches", "scratchy", "flimsy", "fragile", "unreliable",
    "laggy", "sluggish", "stutter", "freeze", "freezes", "freezing", "crash", "crashes",
    "drop", "drops", "leak", "leaks", "rattly", "loose", "slow",
    "dies", "dead", "death", "dying", "drain", "drains", "drained",
}


def _extract_target(prompt: str) -> str:
    m = re.search(r'["\u201c](.*?)["\u201d]', prompt, re.S)
    if m:
        return m.group(1)
    m = re.search(r'(?::|following (?:text|sentence|passage|review|paragraph|article))\s*(.*)', prompt, re.I | re.S)
    return m.group(1) if m else prompt


def solve_sentiment(prompt: str) -> tuple[str, float]:
    target = _extract_target(prompt)
    words = re.findall(r"[a-z]+", target.lower())
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return "neutral. The text contains no strong sentiment indicators.", 0.5
    score = (pos - neg) / total
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "mixed")
    mixed_signal = pos > 0 and neg > 0
    evidence = []
    top_pos = [w for w in words if w in _POS_WORDS][:3]
    top_neg = [w for w in words if w in _NEG_WORDS][:3]
    if top_pos:
        evidence.append(f"positive words: {', '.join(top_pos)}")
    if top_neg:
        evidence.append(f"negative words: {', '.join(top_neg)}")
    return f"The sentiment is {label}. Key indicators: {'; '.join(evidence)}.", min(0.5 + total * 0.1 + (0.15 if mixed_signal else 0.0), 1.0)


_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
           r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_PAT = re.compile(
    rf"\b(?:\d{{1,2}}[\/\-\.]\d{{1,2}}[\/\-\.]\d{{2,4}}"
    rf"|\d{{4}}[\/\-\.]\d{{1,2}}[\/\-\.]\d{{1,2}}"
    rf"|{_MONTHS}\s+\d{{1,2}}(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}\s+{_MONTHS}\s*,?\s*\d{{4}}"
    rf"|\b\d{{4}}\b(?=\s|$|[,\.]))\b", re.I)
_ORG_SUFFIXES = re.compile(
    r"\b([A-Z][A-Za-z&\-]*(?:\s+[A-Z][A-Za-z&\-]*)*"
    r"\s+(?:Inc|Corp|Ltd|LLC|Co|Company|Foundation|Institute|University|Association"
    r"|Organization|Agency|Department|Commission|Bank|Group|International)\.?)\b")
_LOCATION_PREPS = re.compile(
    r"\b(?:in|at|from|near|to|across|through|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_NAME_STOPS = {
    "the", "a", "an", "this", "that", "it", "its", "i", "we", "you", "he",
    "she", "they", "is", "are", "was", "were", "has", "have", "had",
    "if", "when", "where", "what", "which", "who", "how", "why",
    "for", "with", "from", "by", "on", "in", "at", "to", "of",
    "and", "or", "but", "not", "no", "so",
    "identify", "extract", "list", "find", "name", "named", "entities",
}


def solve_ner(prompt: str) -> tuple[str, float]:
    target = _extract_target(prompt)
    entities: dict[str, list[str]] = {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": []}
    for m in _DATE_PAT.finditer(target):
        v = m.group(0).strip()
        if v and v not in entities["DATE"]:
            entities["DATE"].append(v)
    org_spans: set[tuple[int, int]] = set()
    for m in _ORG_SUFFIXES.finditer(target):
        v = m.group(1).strip()
        if v and v not in entities["ORGANIZATION"]:
            entities["ORGANIZATION"].append(v)
            org_spans.add((m.start(), m.end()))
    loc_spans: set[tuple[int, int]] = set()
    for m in _LOCATION_PREPS.finditer(target):
        v = m.group(1).strip()
        if v and v not in entities["LOCATION"] and v.lower() not in _NAME_STOPS:
            entities["LOCATION"].append(v)
            loc_spans.add((m.start(1), m.end(1)))
    used = org_spans | loc_spans
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", target):
        v = m.group(1).strip()
        if any(s <= m.start() < e or s < m.end() <= e for s, e in used):
            continue
        if v.lower() not in _NAME_STOPS and v not in entities["PERSON"]:
            entities["PERSON"].append(v)
    parts = [f"{k}: {', '.join(v)}" for k, v in entities.items() if v]
    total = sum(len(v) for v in entities.values())
    if total == 0:
        return "No named entities found.", 0.3
    return "; ".join(parts), min(0.6 + total * 0.1, 1.0)


def solve_summarization(prompt: str) -> tuple[str, float]:
    target = _extract_target(prompt)
    if not target or target == prompt:
        m = re.search(r'(?:summarize|summary of)\s*(.*)', prompt, re.I | re.S)
        target = m.group(1) if m else prompt
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', target) if s.strip()]
    if len(sentences) <= 2:
        return target.strip(), 0.85
    words = re.findall(r"[a-z]+", target.lower())
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "it", "its", "this", "that",
            "with", "from", "by", "as", "be", "has", "have", "had"}
    freq = Counter(w for w in words if w not in stop and len(w) > 2)
    scored = []
    for i, sent in enumerate(sentences):
        sw = re.findall(r"[a-z]+", sent.lower())
        score = sum(freq.get(w, 0) for w in sw)
        if i == 0: score += 2.0
        elif i == len(sentences) - 1: score += 1.0
        scored.append((score, i, sent))
    n = max(1, min(5, len(sentences) // 3))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]
    top.sort(key=lambda x: x[1])
    result = " ".join(s for _, _, s in top)
    # If the prompt asks for exactly one sentence and we output one sentence, boost confidence
    if re.search(r"exactly one sentence|one sentence only|single sentence", prompt, re.I) and len(top) == 1:
        return result, 0.9
    return result, min(0.7 + len(sentences) * 0.02, 0.95)


_BUILTIN_NAMES = set(dir(builtins))

def solve_code_debug(prompt: str) -> tuple[str, float]:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", prompt, re.S)
    if m:
        code = m.group(1)
    else:
        m2 = re.search(r"((?:def |class |import |from |if |for |while |print\().*)", prompt, re.S)
        code = m2.group(1) if m2 else ""
    if not code.strip():
        return "No code found to debug.", 0.2
    issues: list[str] = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error on line {e.lineno}: {e.msg}")
        if "expected ':'" in str(e.msg):
            issues.append("Suggestion: Add a colon ':' at the end of the statement.")
        elif "invalid syntax" in str(e.msg) and e.text:
            issues.append(f"Near: {e.text.strip()}")
        return "Issues found:\n" + "\n".join(f"- {x}" for x in issues), 0.8
    try:
        def _walk(tbl: symtable.SymbolTable) -> None:
            for sym in tbl.get_symbols():
                n = sym.get_name()
                if sym.is_referenced() and not sym.is_assigned() and not sym.is_imported() and not sym.is_parameter():
                    if n not in _BUILTIN_NAMES and n not in {"self", "cls"}:
                        issues.append(f"Possibly undefined name: '{n}'")
            for child in tbl.get_children():
                _walk(child)
        _walk(symtable.symtable(code, "<code>", "exec"))
    except Exception:
        pass
    for i, line in enumerate(code.split("\n"), 1):
        if re.match(r"^\s*if\b.*[^=!<>]=[^=]", line) and "==" not in line and "!=" not in line:
            issues.append(f"Line {i}: Possible assignment '=' in condition, did you mean '=='?")
        if re.match(r"\s*def\s+\w+\(.*=\s*\[\]", line) or re.match(r"\s*def\s+\w+\(.*=\s*\{\}", line):
            issues.append(f"Line {i}: Mutable default argument (use None instead).")
    if not issues:
        return "No issues found. The code appears syntactically correct with no obvious bugs.", 0.7
    return "Issues found:\n" + "\n".join(f"- {x}" for x in issues), 0.85


def solve_logical(prompt: str) -> tuple[str, float]:
    lower = prompt.lower()
    syl = re.search(
        r"(?:if )?all\s+(\w+)\s+are\s+(\w+).*?(\w+)\s+is\s+(?:a |an )?(\w+).*?"
        r"(?:is\s+(\w+)\s+(?:a |an )?(\w+))", lower, re.S)
    if syl:
        cat_a, cat_b, inst, inst_cat = syl.group(1), syl.group(2), syl.group(3), syl.group(4)
        q_subj, q_pred = syl.group(5), syl.group(6)
        if inst_cat == cat_a and q_pred == cat_b:
            return f"Yes, {q_subj} is {q_pred} because all {cat_a} are {cat_b} and {inst} is {inst_cat}.", 0.9
        if inst_cat == cat_a:
            return "Cannot be determined from the given premises.", 0.8
    if "always lies" in lower and "always tells the truth" in lower:
        return ("To solve this, ask: 'If I asked the other person which door is safe, "
                "what would they say?' Then choose the opposite."), 0.8
    seq = re.search(r"(?:what (?:comes|is) next|next (?:number|term|in the sequence)).*?(\d+(?:\s*,\s*\d+){2,})", lower)
    if seq:
        nums = [int(x.strip()) for x in seq.group(1).split(",")]
        if len(nums) >= 3:
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            if len(set(diffs)) == 1:
                return f"The next number is {nums[-1] + diffs[0]}. This is an arithmetic sequence with common difference {diffs[0]}.", 0.95
            if all(nums[i] != 0 for i in range(len(nums)-1)):
                ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1)]
                if len(set(ratios)) == 1:
                    nv = nums[-1] * ratios[0]
                    return f"The next number is {int(nv) if nv == int(nv) else nv}. This is a geometric sequence with common ratio {ratios[0]}.", 0.95
    # handles "each own/painted/drove/built/chose a different X" + one positive + one negation + "who owns Y?"
    # ceiling: multiple negations or absent positive assignment — defer to T1.
    verbs = r"owns?|painted|painted|drove|built|chose|picked"
    has = re.search(rf"\b(?!not\b)(\w+)\s+(?:{verbs})\s+the\s+(\w+)\b", lower)
    not_has = re.search(rf"\b(\w+)\s+does\s+not\s+(?:{verbs})\s+(?:the\s+)?(\w+)\b", lower)
    who = re.search(r"who\s+(?:owns?|painted|drove|built|chose|picked)\s+(?:the\s+)?(\w+)\??\s*$", lower)
    items_all = re.search(r"each\s+(?:own|painted|drove|built|chose|picked)\s+(?:a\s+)?different\s+\w+:?\s*([^.?]+)", lower)
    if has and who and items_all:
        items = [w.strip() for w in re.split(r"[,\s]+(?:and\s+)?", items_all.group(1)) if w.strip()]
        assigned, owner, not_item = has.group(1).lower(), has.group(2).strip(), None
        if not_has:
            not_item = not_has.group(2).strip()
        target = who.group(1).strip().rstrip("?.")
        names = [w for w in re.findall(r"\b([A-Z][a-z]+)\b", prompt)
                 if w.lower() not in {"the", "who", "owns", "does", "not", "each", "three", "two", "four", "five"}
                 and w.lower() != assigned]
        candidates = [n for n in dict.fromkeys(names) if n.lower() != assigned]  # dedupe, keep order
        if assigned == target or (not_item and target == not_item):
            if not_item and target == not_item:
                free = [n for n in candidates if n.lower() != not_has.group(1)]
                if len(free) == 1:
                    return f"{free[0]} owns the {target}.", 0.85
            if assigned == target and not not_has:
                return f"{assigned.title()} owns the {target}.", 0.9
        if assigned != target:
            if not_has and not_has.group(1) != assigned:
                third = [n for n in candidates if n.lower() != not_has.group(1)]
                if len(third) == 1:
                    return f"{third[0]} owns the {target}.", 0.85
            elif not not_has and len(candidates) == 1:
                return f"{candidates[0]} owns the {target}.", 0.85
    return "", 0.3


_FACTUAL_DB: dict[str, str] = {
    "capital of australia": "Canberra", "capital of france": "Paris",
    "capital of japan": "Tokyo", "capital of germany": "Berlin",
    "capital of italy": "Rome", "capital of spain": "Madrid",
    "capital of canada": "Ottawa", "capital of brazil": "Brasilia",
    "capital of india": "New Delhi", "capital of china": "Beijing",
    "capital of russia": "Moscow", "capital of united kingdom": "London",
    "capital of uk": "London", "capital of united states": "Washington, D.C.",
    "capital of usa": "Washington, D.C.", "capital of mexico": "Mexico City",
    "capital of south korea": "Seoul", "capital of egypt": "Cairo",
    "capital of south africa": "Pretoria", "capital of argentina": "Buenos Aires",
    "capital of turkey": "Ankara", "capital of indonesia": "Jakarta",
    "capital of philippines": "Manila", "capital of thailand": "Bangkok",
    "capital of greece": "Athens", "capital of portugal": "Lisbon",
    "capital of netherlands": "Amsterdam", "capital of sweden": "Stockholm",
    "capital of norway": "Oslo", "capital of finland": "Helsinki",
    "capital of denmark": "Copenhagen", "capital of poland": "Warsaw",
    "capital of austria": "Vienna", "capital of switzerland": "Bern",
    "capital of ireland": "Dublin", "capital of belgium": "Brussels",
    "capital of new zealand": "Wellington", "capital of saudi arabia": "Riyadh",
    "capital of iran": "Tehran", "capital of iraq": "Baghdad",
    "capital of israel": "Jerusalem", "capital of kenya": "Nairobi",
    "capital of nigeria": "Abuja", "capital of morocco": "Rabat",
    "capital of peru": "Lima", "capital of chile": "Santiago",
    "capital of colombia": "Bogota", "capital of venezuela": "Caracas",
    "largest planet": "Jupiter", "smallest planet": "Mercury",
    "tallest mountain": "Mount Everest at 8,849 meters (29,032 feet)",
    "largest ocean": "The Pacific Ocean",
    "smallest ocean": "The Arctic Ocean",
    "longest river": "The Nile River",
    "largest desert": "The Antarctic Desert",
    "largest country": "Russia",
    "smallest country": "Vatican City",
    "deepest ocean": "The Mariana Trench in the Pacific Ocean",
    # Bodies of water near capitals (caps + water combo queries)
    "body of water near canberra": "Lake Burley Griffin (artificial lake), and the Molonglo River",
    "body of water near australia capital": "Lake Burley Griffin",
    "body of water near tokyo": "Tokyo Bay (part of the Pacific Ocean), and the Sumida River",
    "body of water near japan capital": "Tokyo Bay",
    "body of water near paris": "The Seine River",
    "body of water near france capital": "The Seine River",
    "body of water near london": "The River Thames",
    "body of water near united kingdom capital": "The River Thames",
    "body of water near berlin": "The Spree River, and it connects to the Havel River",
    "body of water near germany capital": "The Spree River",
    "body of water near rome": "The Tiber River",
    "body of water near italy capital": "The Tiber River",
    "body of water near ottawa": "The Ottawa River, and it sits at the confluence of three rivers",
    "body of water near canada capital": "The Ottawa River",
    "body of water near brasilia": "Lake Paranoa (artificial lake), and rivers feeding it",
    "body of water near brazil capital": "Lake Paranoa",
    "body of water near washington": "The Potomac River, and the Anacostia River",
    "body of water near united states capital": "The Potomac River",
    "body of water near new delhi": "The Yamuna River (a tributary of the Ganges)",
    "body of water near india capital": "The Yamuna River",
    "body of water near beijing": "Kunming Lake, and the Yongding River nearby",
    "body of water near china capital": "Kunming Lake",
    "body of water near moscow": "The Moscow River",
    "body of water near russia capital": "The Moscow River",
    "body of water near manila": "Manila Bay (part of the South China Sea), and the Pasig River",
    "body of water near philippines capital": "Manila Bay",
    "body of water near seoul": "The Han River",
    "body of water near south korea capital": "The Han River",
    "body of water near cairo": "The Nile River",
    "body of water near egypt capital": "The Nile River",
    "body of water near athens": "The Saronic Gulf (part of the Aegean Sea)",
    "body of water near greece capital": "The Saronic Gulf",
    "body of water near amsterdam": "The Amstel River, and the IJ bay (part of Lake IJssel)",
    "body of water near netherlands capital": "The Amstel River",
    "body of water near stockholm": "The Baltic Sea, and Lake Malaren",
    "body of water near sweden capital": "The Baltic Sea",
    "body of water near bangkok": "The Chao Phraya River",
    "body of water near thailand capital": "The Chao Phraya River",
    "body of water near jakarta": "The Java Sea, and the Ciliwung River",
    "body of water near indonesia capital": "The Java Sea",
    "body of water near wellington": "Wellington Harbour (part of the Pacific Ocean)",
    "body of water near new zealand capital": "Wellington Harbour",
    "body of water near buenos aires": "The Rio de la Plata (a river estuary)",
    "body of water near argentina capital": "The Rio de la Plata",
    "body of water near istanbul": "The Bosphorus Strait, and the Sea of Marmara",
    "body of water near turkey capital": "The Kizilirmak River ( Ankara is inland)",
    "body of water near nairobi": "The Nairobi River, and it is near Lake Nairobi",
    "body of water near kenya capital": "The Nairobi River",
    "body of water near lima": "The Pacific Ocean, and the Rimac River",
    "body of water near peru capital": "The Rimac River",
    "body of water near dublin": "The River Liffey, and Dublin Bay (part of the Irish Sea)",
    "body of water near ireland capital": "The River Liffey",
    "body of water near oslo": "The Oslo Fjord",
    "body of water near norway capital": "The Oslo Fjord",
    "body of water near helsinki": "The Gulf of Finland (part of the Baltic Sea)",
    "body of water near finland capital": "The Gulf of Finland",
    "body of water near copenhagen": "The Oresund strait",
    "body of water near denmark capital": "The Oresund strait",
    "body of water near vienna": "The Danube River",
    "body of water near austria capital": "The Danube River",
    "body of water near bern": "The Aare River, and it sits on a peninsula",
    "body of water near switzerland capital": "The Aare River",
    "body of water near warsaw": "The Vistula River",
    "body of water near poland capital": "The Vistula River",
    "body of water near lisbon": "The Tagus River, and the Atlantic Ocean",
    "body of water near portugal capital": "The Tagus River",
}


def solve_factual(prompt: str) -> tuple[str, float]:
    lower = prompt.lower().strip().rstrip("?!. ")
    cap = re.search(r"capital of\s+(\w[\w\s]*\w)", lower)
    water = re.search(r"(?:body of water|river|lake|sea|ocean|bay)", lower)
    if cap and water:
        country = cap.group(1).strip()
        if country.startswith("the "):
            country = country[4:]
        capital = _FACTUAL_DB.get(f"capital of {country}")
        w = _FACTUAL_DB.get(f"body of water near {country} capital")
        if capital and w:
            return f"The capital of {country.title()} is {capital}. Nearby body of water: {w}.", 0.9
        if capital:
            return f"The capital of {country.title()} is {capital}.", 0.7
    elif cap:
        country = cap.group(1).strip()
        if country.startswith("the "):
            country = country[4:]
        capital = _FACTUAL_DB.get(f"capital of {country}")
        if capital:
            return f"The capital of {country.title()} is {capital}.", 0.9
    for key, val in _FACTUAL_DB.items():
        if key in lower:
            return val, 0.85
    return "UNKNOWN", 0.3


_SOLVERS = {
    "math": solve_math, "sentiment": solve_sentiment, "ner": solve_ner,
    "summarization": solve_summarization, "code_debug": solve_code_debug,
    "logical": solve_logical, "factual": solve_factual,
}
_CONF_THRESHOLD = 0.8


def verify_code_debug(fixed_code: str) -> bool:
    # ponytail: runs solver-fixed code in a stripped namespace to catch
    # logic bugs (not just syntax). max-like funcs get multi-input asserts;
    # others only crash-detect — true logic check needs a spec we don't have.
    # Safe builtins only: blocks open/eval/exec/import, keeps math helpers.
    if not fixed_code.strip():
        return True
    _safe_builtins = {
        "max": max, "min": min, "len": len, "sorted": sorted, "sum": sum,
        "abs": abs, "range": range, "int": int, "float": float, "round": round,
        "list": list, "dict": dict, "tuple": tuple, "str": str, "set": set,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "reversed": reversed, "any": any, "all": all, "bool": bool,
    }
    safe_globals = {"__builtins__": _safe_builtins}
    try:
        exec(compile(fixed_code, "<string>", "exec"), safe_globals)
        funcs = [v for k, v in safe_globals.items()
                 if callable(v) and not k.startswith("__")]
        if not funcs:
            return True
        func = funcs[0]
        name = func.__name__.lower() if hasattr(func, "__name__") else ""
        if "max" in name or "largest" in name or "greatest" in name:
            for args, expect in [([1, 3, 2], 3), ([5, 5, 1], 5), ([-1, -3], -1)]:
                try:
                    if func(args) != expect:
                        return False
                except Exception:
                    return False
            return True
        try:
            func(1)
        except TypeError:
            try:
                func("x")
            except Exception:
                return False
        except Exception:
            return False
        return True
    except Exception:
        return False


def verify_math(result: str, prompt: str) -> bool:
    try:
        val = float(result)
        if val != val or val == float('inf') or val == float('-inf'):
            return False
        if abs(val) >= 1e15:
            return False
        return True
    except Exception:
        return False


def dispatch(prompt: str) -> dict:
    if _track1_deterministic_answer is not None:
        answer = _track1_deterministic_answer(prompt)
        if answer:
            category = (
                _track1_classify_domain(prompt)
                if _track1_classify_domain is not None
                else classify(prompt)
            )
            return {
                "tier": "T0",
                "category": category,
                "answer": answer,
                "confidence": 1.0,
                "tokens": 0,
            }

    category = classify(prompt)
    solver = _SOLVERS.get(category)
    if solver is not None:
        answer, conf = solver(prompt)
        if answer and conf >= _CONF_THRESHOLD:
            if category == "math":
                if not verify_math(answer, prompt):
                    return {"tier": "T1", "category": category, "answer": "", "confidence": 0.0, "prompt": prompt}
            elif category == "code_debug" and answer.startswith("No issues found"):
                m = re.search(r"```(?:python)?\s*\n(.*?)```", prompt, re.S)
                if m:
                    code = m.group(1)
                else:
                    m2 = re.search(r"((?:def |class |import |from |if |for |while |print\().*)", prompt, re.S)
                    code = m2.group(1) if m2 else ""
                if not verify_code_debug(code):
                    return {"tier": "T1", "category": category, "answer": "", "confidence": 0.0, "prompt": prompt}
            return {"tier": "T0", "category": category, "answer": answer, "confidence": conf, "tokens": 0}
    return {"tier": "T1", "category": category, "answer": "", "confidence": 0.0, "prompt": prompt}


_DEMO_PROMPTS = [
    "What is the capital of Australia, and what body of water is it near?",
    "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?",
    'Analyze the sentiment of this review: "The movie was absolutely wonderful, with brilliant acting and a heartwarming story."',
    'Summarize the following text: "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance through experience. ML algorithms can be categorized into supervised learning, unsupervised learning, and reinforcement learning. Each approach has distinct use cases and advantages."',
    'Identify the named entities in: "John Smith visited the United Nations headquarters in New York on January 15, 2024."',
    'Debug this code:\n```python\ndef greet(name):\n    message = "Hello, " + name\n    print(mesage)\n```',
    "What comes next in the sequence: 2, 6, 18, 54?",
    "Write a function to reverse a linked list.",
]


def demo() -> None:
    print("=" * 60)
    print("ROUTER DEMO")
    print("=" * 60)
    for i, prompt in enumerate(_DEMO_PROMPTS, 1):
        r = dispatch(prompt)
        cat, tier, conf = r["category"], r["tier"], r.get("confidence", 0.0)
        ans = r.get("answer", "")
        print(f"\n[{i}] {prompt[:75]}{'...' if len(prompt)>75 else ''}")
        print(f"    cat={cat}  tier={tier}  conf={conf:.2f}")
        if tier == "T0":
            print(f"    answer: {ans[:100]}{'...' if len(ans)>100 else ''}")
        else:
            print(f"    -> escalate to local model (T1)")
    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    demo()
