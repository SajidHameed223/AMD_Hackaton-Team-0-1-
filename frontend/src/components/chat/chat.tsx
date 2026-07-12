"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import type { Route } from "@/lib/types";
import { Avatar } from "../core";

/* ---- RouteBadge — the router's verdict, worn by each assistant turn ---- */
export function RouteBadge({
  route,
  model,
  latencyMs,
}: {
  route: Route | "routing";
  model?: string;
  latencyMs?: number;
}) {
  if (route === "routing") {
    return (
      <span className="o1-route o1-route--routing">
        <span className="o1-route__icon" aria-hidden>
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
            <path
              d="M14 8a6 6 0 1 1-1.76-4.24"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        </span>
        routing…
      </span>
    );
  }
  return (
    <span className={`o1-route o1-route--${route}`}>
      <span className="o1-route__icon" aria-hidden>
        {route === "local" ? (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
            <path
              d="M8.8 1.5 3.5 9h3.4l-.7 5.5L11.5 7H8.1l.7-5.5Z"
              fill="currentColor"
            />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
            <path
              d="M4.6 12.5a3.1 3.1 0 0 1-.4-6.18 4 4 0 0 1 7.75-.9 2.9 2.9 0 0 1-.55 5.73l-6.8 1.35Z"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinejoin="round"
              fill="none"
              transform="translate(0 -1.2)"
            />
          </svg>
        )}
      </span>
      {route}
      {model ? ` · ${model}` : null}
      {latencyMs !== undefined && (
        <span className="o1-route__latency">· {latencyMs} ms</span>
      )}
    </span>
  );
}

/* ---- TypingIndicator — three cyan dots riding a wave ---- */
export function TypingIndicator() {
  return (
    <span className="o1-typing" role="status" aria-label="O(1) is thinking">
      <span className="o1-typing__dot" />
      <span className="o1-typing__dot" />
      <span className="o1-typing__dot" />
    </span>
  );
}

/* ---- StreamingText — blinking cyan caret on in-flight output ---- */
export function StreamingText({
  active = true,
  children,
}: {
  active?: boolean;
  children: ReactNode;
}) {
  return (
    <span>
      {children}
      {active && <span className="o1-stream__cursor" aria-hidden />}
    </span>
  );
}

/* ---- Message — one conversation turn ---- */
export function Message({
  role,
  meta,
  actions,
  children,
}: {
  role: "user" | "assistant";
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  if (role === "user") {
    return (
      <div className="o1-msg o1-msg--user">
        <div className="o1-msg__body">{children}</div>
      </div>
    );
  }
  return (
    <div className="o1-msg">
      <Avatar kind="assistant" />
      <div className="o1-msg__body">
        <div className="o1-msg__meta">
          <span className="o1-msg__author">O(1)</span>
          {meta}
        </div>
        {children}
        {actions && <div className="o1-msg__actions">{actions}</div>}
      </div>
    </div>
  );
}

/* ---- ChatWelcome — big display greeting with a gradient word ---- */
export function ChatWelcome({
  title,
  subtitle,
  suggestions,
  onPick,
}: {
  title: ReactNode;
  subtitle?: string;
  suggestions?: string[];
  onPick?: (s: string) => void;
}) {
  return (
    <div className="o1-welcome">
      <h1 className="o1-welcome__title">{title}</h1>
      {subtitle && <p className="o1-welcome__sub">{subtitle}</p>}
      {suggestions && suggestions.length > 0 && (
        <div className="o1-welcome__chips">
          {suggestions.map((s) => (
            <button
              key={s}
              className="o1-welcome__chip"
              onClick={() => onPick?.(s)}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- ChatInput — auto-growing composer with the gradient send pearl ---- */
export function ChatInput({
  onSend,
  disabled = false,
  placeholder,
  footStart,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  footStart?: ReactNode;
}) {
  const [value, setValue] = useState("");
  const [attachment, setAttachment] = useState<{ name: string; text: string } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [readingFile, setReadingFile] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const readAttachment = useCallback(async (file: File): Promise<string> => {
    if (file.size > 5 * 1024 * 1024) throw new Error("Files must be 5 MB or smaller.");
    if (file.name.toLowerCase().endsWith(".txt") || file.type === "text/plain") {
      return (await file.text()).slice(0, 50_000);
    }
    if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
      const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
      const pdfOptions = {
        data: new Uint8Array(await file.arrayBuffer()),
        disableWorker: true,
      };
      const document = await pdfjs.getDocument(
        pdfOptions as Parameters<typeof pdfjs.getDocument>[0],
      ).promise;
      const pages: string[] = [];
      for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
        const page = await document.getPage(pageNumber);
        const content = await page.getTextContent();
        pages.push(content.items.map((item) => ("str" in item ? item.str : "")).join(" "));
      }
      return pages.join("\n\n").slice(0, 50_000);
    }
    throw new Error("Attach a .txt or .pdf file.");
  }, []);

  const onFileChange = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setFileError(null);
    setReadingFile(true);
    try {
      const text = (await readAttachment(file)).trim();
      if (!text) throw new Error("That file does not contain readable text.");
      setAttachment({ name: file.name, text });
    } catch (error) {
      setAttachment(null);
      setFileError(error instanceof Error ? error.message : "Could not read that file.");
    } finally {
      setReadingFile(false);
    }
  }, [readAttachment]);

  const autogrow = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const submit = useCallback(() => {
    const text = value.trim();
    if ((!text && !attachment) || disabled || readingFile) return;
    const prompt = attachment
      ? (text || "Please read the attached file and respond.") + "\n\n[Attached file: " + attachment.name + "]\n" + attachment.text
      : text;
    onSend(prompt);
    setValue("");
    setAttachment(null);
    setFileError(null);
    requestAnimationFrame(() => {
      const el = ref.current;
      if (el) el.style.height = "auto";
    });
  }, [value, attachment, disabled, onSend, readingFile]);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="o1-input">
      <div className="o1-input__row">
        <button
          type="button"
          className="o1-input__attach"
          onClick={() => fileRef.current?.click()}
          disabled={disabled || readingFile}
          aria-label="Attach a text or PDF file"
          title="Attach a .txt or .pdf file"
        >
          +
        </button>
        <input
          ref={fileRef}
          className="o1-sr-only"
          type="file"
          accept=".txt,.pdf,text/plain,application/pdf"
          onChange={onFileChange}
          tabIndex={-1}
        />
        <textarea
          id="o1-chat-input"
          name="message"
          ref={ref}
          className="o1-input__field"
          rows={1}
          value={value}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onInput={autogrow}
          onKeyDown={onKeyDown}
          aria-label="Message O(1)"
        />
        <button
          className="o1-input__send"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send"
        >
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden>
            <path
              d="M10 15.5v-11m0 0L5 9.2m5-4.7 5 4.7"
              stroke="currentColor"
              strokeWidth="2.1"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {(attachment || fileError || readingFile) && (
        <div className="o1-input__attachment" role="status">
          {readingFile && <span>Reading file…</span>}
          {!readingFile && attachment && (
            <>
              <span title={attachment.name}>📎 {attachment.name}</span>
              <button type="button" onClick={() => setAttachment(null)} aria-label="Remove attachment">Remove</button>
            </>
          )}
          {!readingFile && fileError && <span className="o1-input__file-error">{fileError}</span>}
        </div>
      )}
      <div className="o1-input__foot">
        <div className="o1-input__status">
          {footStart ?? <span />}
          <AutoRouteBadge />
        </div>
        <span className="o1-input__hint">
          Enter to send · Shift+Enter for a new line
        </span>
      </div>
    </div>
  );
}

function AutoRouteBadge() {
  return (
    <span className="o1-auto-route" aria-label="Automatic model routing enabled">
      Auto
    </span>
  );
}

/* ---- ChatWindow — rounded sand-white canvas + scrolling column ---- */
export function ChatWindow({
  composer,
  children,
  scrollRef,
}: {
  composer: ReactNode;
  children: ReactNode;
  scrollRef?: React.Ref<HTMLDivElement>;
}) {
  return (
    <div className="o1-chat">
      <div className="o1-chat__scroll" ref={scrollRef}>
        <div className="o1-chat__inner">{children}</div>
      </div>
      <div className="o1-chat__composer">{composer}</div>
    </div>
  );
}

/* Keep the conversation pinned to the newest turn while streaming. */
export function useAutoScroll(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      pinned.current =
        el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [dep]);

  return ref;
}
