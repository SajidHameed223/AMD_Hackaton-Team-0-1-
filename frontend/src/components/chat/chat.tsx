"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import type { Route, RoutePreference } from "@/lib/types";
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
  routePreference,
  onRoutePreferenceChange,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  footStart?: ReactNode;
  routePreference: RoutePreference;
  onRoutePreferenceChange: (preference: RoutePreference) => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const autogrow = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const submit = useCallback(() => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    requestAnimationFrame(() => {
      const el = ref.current;
      if (el) el.style.height = "auto";
    });
  }, [value, disabled, onSend]);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="o1-input">
      <div className="o1-input__row">
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
      <div className="o1-input__foot">
        <div className="o1-input__status">
          {footStart ?? <span />}
          <RoutePreferenceSwitch
            value={routePreference}
            onChange={onRoutePreferenceChange}
            disabled={disabled}
          />
        </div>
        <span className="o1-input__hint">
          Enter to send · Shift+Enter for a new line
        </span>
      </div>
    </div>
  );
}

function RoutePreferenceSwitch({
  value,
  onChange,
  disabled,
}: {
  value: RoutePreference;
  onChange: (preference: RoutePreference) => void;
  disabled?: boolean;
}) {
  const items: { id: RoutePreference; label: string; route?: Route }[] = [
    { id: "auto", label: "Auto" },
    { id: "local", label: "Local", route: "local" },
    { id: "cloud", label: "Cloud", route: "cloud" },
  ];

  return (
    <div className="o1-model-switch" role="radiogroup" aria-label="Model route">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="radio"
          aria-checked={value === item.id}
          className={`o1-model-switch__item${
            value === item.id ? " o1-model-switch__item--active" : ""
          }${item.route ? ` o1-model-switch__item--${item.route}` : ""}`}
          onClick={() => onChange(item.id)}
          disabled={disabled}
        >
          {item.label}
        </button>
      ))}
    </div>
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
