"use client";

import { useCallback, useRef, useState, type ReactNode } from "react";

/* ---- CopyButton — one-click copy with a transient confirmation ---- */
export function CopyButton({
  text,
  surface = "dark",
}: {
  text: string;
  surface?: "dark" | "light";
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard API unavailable (non-secure context) — fall back
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1800);
  }, [text]);

  const cls = [
    "o1-copy",
    surface === "light" ? "o1-copy--light" : "",
    copied ? "o1-copy--copied" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button className={cls} onClick={copy} aria-live="polite">
      {copied ? (
        <>
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden>
            <path
              d="M2.5 8.5 6 12l7.5-8"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Copied
        </>
      ) : (
        <>
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden>
            <rect
              x="5.5"
              y="5.5"
              width="8"
              height="8"
              rx="2"
              stroke="currentColor"
              strokeWidth="1.6"
            />
            <path
              d="M10.5 3.5v-1a1.5 1.5 0 0 0-1.5-1.5H4A1.5 1.5 0 0 0 2.5 2.5V8A1.5 1.5 0 0 0 4 9.5h1"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
          Copy
        </>
      )}
    </button>
  );
}

export function RetryButton({
  onRetry,
  disabled = false,
  surface = "dark",
}: {
  onRetry: () => void;
  disabled?: boolean;
  surface?: "dark" | "light";
}) {
  const cls = [
    "o1-copy",
    "o1-retry",
    surface === "light" ? "o1-copy--light" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button className={cls} onClick={onRetry} disabled={disabled}>
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path
          d="M13.2 7.1a5.2 5.2 0 1 0-1.5 4.2M13.2 7.1V3.8m0 3.3H9.9"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      Retry
    </button>
  );
}

/* ---- mini syntax highlighter → o1-tok-* spans ---- */
const KEYWORDS = new Set([
  // shared
  "if", "else", "for", "while", "return", "break", "continue", "class",
  "import", "from", "in", "new", "try", "catch", "finally", "throw",
  "switch", "case", "default", "do", "typeof", "delete", "void", "this",
  // python
  "def", "elif", "lambda", "pass", "raise", "with", "as", "yield", "global",
  "nonlocal", "assert", "del", "not", "and", "or", "is", "None", "True", "False",
  // js / ts
  "const", "let", "var", "function", "async", "await", "export", "extends",
  "implements", "interface", "type", "enum", "null", "undefined", "true",
  "false", "static", "public", "private", "readonly",
  // sql-ish / misc
  "select", "where", "struct", "fn", "match", "mut", "pub",
]);

const HASH_COMMENT_LANGS = /^(py|python|rb|ruby|sh|bash|shell|zsh|yaml|yml|toml|r)$/;
const SLASH_COMMENT_LANGS =
  /^(js|jsx|ts|tsx|javascript|typescript|c|h|cpp|cc|java|go|rust|rs|kt|kotlin|swift|cs|csharp|scala|php|json5)$/;

function tokenRegex(language?: string): RegExp {
  const lang = (language ?? "").toLowerCase();
  // `//` is integer division in Python — only treat it as a comment in
  // C-family languages; `#` is not a comment in C-family.
  const comment = HASH_COMMENT_LANGS.test(lang)
    ? "#[^\\n]*"
    : SLASH_COMMENT_LANGS.test(lang)
      ? "\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/"
      : "#[^\\n]*|\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/";
  return new RegExp(
    `(${comment})|("""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*'|\`(?:\\\\.|[^\`\\\\])*\`)|(\\b\\d(?:[\\d_]*\\.?[\\d_]*)(?:e[+-]?\\d+)?\\b)|([A-Za-z_][A-Za-z0-9_]*)`,
    "g",
  );
}

export function highlight(code: string, language?: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const m of code.matchAll(tokenRegex(language))) {
    const idx = m.index ?? 0;
    if (idx > last) out.push(code.slice(last, idx));
    const [text, comment, str, num, word] = m;
    if (comment) out.push(<span key={key++} className="o1-tok-c">{text}</span>);
    else if (str) out.push(<span key={key++} className="o1-tok-s">{text}</span>);
    else if (num) out.push(<span key={key++} className="o1-tok-n">{text}</span>);
    else if (word) {
      if (KEYWORDS.has(word)) {
        out.push(<span key={key++} className="o1-tok-k">{text}</span>);
      } else if (code.slice(idx + text.length).match(/^\s*\(/)) {
        out.push(<span key={key++} className="o1-tok-f">{text}</span>);
      } else {
        out.push(text);
      }
    }
    last = idx + text.length;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

/* ---- CodeBlock — the deep-sea well, the one dark surface ---- */
export function CodeBlock({
  code,
  language,
}: {
  code: string;
  language?: string;
}) {
  return (
    <div className="o1-code">
      <div className="o1-code__bar">
        <span className="o1-code__lang">{language || "code"}</span>
        <CopyButton text={code} />
      </div>
      <pre className="o1-code__body">
        <code>{highlight(code, language)}</code>
      </pre>
    </div>
  );
}
