"use client";

import { useId, type CSSProperties, type ReactNode } from "react";

/* ---- O1Provider — every screen hangs off .o1-root ---- */
export function O1Provider({
  children,
  className,
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div className={`o1-root${className ? ` ${className}` : ""}`} style={style}>
      {children}
    </div>
  );
}

/* ---- Logo — the O is an ocean-gradient ring ---- */
export function Logo({ size = 21 }: { size?: number }) {
  const gid = useId();
  return (
    <span className="o1-logo" style={{ fontSize: size }} aria-label="O(1)">
      <svg
        className="o1-logo__o"
        width={size * 1.08}
        height={size * 1.08}
        viewBox="0 0 100 100"
        aria-hidden
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#22cbe8" />
            <stop offset="0.55" stopColor="#06aece" />
            <stop offset="1" stopColor="#0e7490" />
          </linearGradient>
        </defs>
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke={`url(#${gid})`}
          strokeWidth="13"
          strokeLinecap="round"
        />
      </svg>
      <span className="o1-logo__paren">(</span>
      <span className="o1-logo__one">1</span>
      <span className="o1-logo__paren">)</span>
    </span>
  );
}

/* ---- Button ---- */
export function Button({
  variant = "secondary",
  size,
  icon = false,
  className,
  children,
  ...rest
}: {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "lg";
  icon?: boolean;
  className?: string;
  children?: ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const cls = [
    "o1-btn",
    `o1-btn--${variant}`,
    size ? `o1-btn--${size}` : "",
    icon ? "o1-btn--icon" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button className={cls} {...rest}>
      {children}
    </button>
  );
}

/* ---- Tabs — pill switcher with a spring-sliding thumb ---- */
export function Tabs({
  items,
  value,
  onChange,
  "aria-label": ariaLabel,
}: {
  items: { id: string; label: string }[];
  value: string;
  onChange: (id: string) => void;
  "aria-label"?: string;
}) {
  const index = Math.max(
    0,
    items.findIndex((it) => it.id === value),
  );
  return (
    <div className="o1-tabs" role="tablist" aria-label={ariaLabel}>
      <span
        className="o1-tabs__thumb"
        style={{
          width: `calc((100% - 8px) / ${items.length})`,
          transform: `translateX(${index * 100}%)`,
        }}
        aria-hidden
      />
      {items.map((it) => (
        <button
          key={it.id}
          role="tab"
          aria-selected={it.id === value}
          className={`o1-tabs__tab${it.id === value ? " o1-tabs__tab--active" : ""}`}
          onClick={() => onChange(it.id)}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

/* ---- Card ---- */
export function Card({
  flush = false,
  lift = false,
  className,
  style,
  children,
}: {
  flush?: boolean;
  lift?: boolean;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}) {
  const cls = [
    "o1-card",
    flush ? "o1-card--flush" : "",
    lift ? "o1-card--lift" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={cls} style={style}>
      {children}
    </div>
  );
}

/* ---- Badge ---- */
export function Badge({
  tone = "sand",
  dot = false,
  children,
}: {
  tone?: "cyan" | "coral" | "sun" | "sand" | "good" | "warning" | "critical";
  dot?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={`o1-badge o1-badge--${tone}`}>
      {dot && <span className="o1-badge__dot" aria-hidden />}
      {children}
    </span>
  );
}

/* ---- Avatar ---- */
export function Avatar({
  kind,
  size,
  label,
}: {
  kind: "user" | "assistant";
  size?: "lg";
  label?: string;
}) {
  const gid = useId();
  return (
    <span
      className={`o1-avatar o1-avatar--${kind}${size ? ` o1-avatar--${size}` : ""}`}
      aria-hidden
    >
      {kind === "assistant" ? (
        <svg width="16" height="16" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="38"
            fill="none"
            stroke={`url(#${gid})`}
            strokeWidth="16"
            strokeLinecap="round"
          />
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#ffffff" />
              <stop offset="1" stopColor="#d7f6fc" />
            </linearGradient>
          </defs>
        </svg>
      ) : (
        (label ?? "You").slice(0, 3)
      )}
    </span>
  );
}

/* ---- Skeleton ---- */
export function Skeleton({
  width,
  height = 14,
  style,
}: {
  width?: string | number;
  height?: string | number;
  style?: CSSProperties;
}) {
  return (
    <span
      className="o1-skeleton"
      style={{ display: "block", width: width ?? "100%", height, ...style }}
      aria-hidden
    />
  );
}

/* ---- WaveDivider — the one decorative beach motif, at most once per view ---- */
export function WaveDivider() {
  return (
    <svg
      className="o1-wave"
      viewBox="0 0 1160 14"
      preserveAspectRatio="none"
      aria-hidden
    >
      <path
        d="M0,7 C29,1 58,1 87,7 C116,13 145,13 174,7 C203,1 232,1 261,7 C290,13 319,13 348,7 C377,1 406,1 435,7 C464,13 493,13 522,7 C551,1 580,1 609,7 C638,13 667,13 696,7 C725,1 754,1 783,7 C812,13 841,13 870,7 C899,1 928,1 957,7 C986,13 1015,13 1044,7 C1073,1 1102,1 1131,7 L1160,7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ---- EmptyState ---- */
export function EmptyState({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="o1-empty">
      <div className="o1-empty__title">{title}</div>
      {children && <div className="o1-empty__body">{children}</div>}
    </div>
  );
}

/* ---- TopBar — brand left, tabs centered, status right ---- */
export function TopBar({
  brand,
  nav,
  side,
}: {
  brand?: ReactNode;
  nav?: ReactNode;
  side?: ReactNode;
}) {
  return (
    <header className="o1-topbar">
      <div className="o1-topbar__brand">{brand ?? <Logo />}</div>
      <nav className="o1-topbar__nav">{nav}</nav>
      <div className="o1-topbar__side">{side}</div>
    </header>
  );
}

/* ---- AppShell ---- */
export function AppShell({
  topBar,
  children,
}: {
  topBar: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="o1-shell">
      {topBar}
      <main className="o1-shell__main">{children}</main>
    </div>
  );
}
