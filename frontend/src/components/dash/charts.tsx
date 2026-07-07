"use client";

import { useMemo, useRef, useState, type ReactNode } from "react";
import type { StatDelta } from "@/lib/types";
import { Card } from "../core";

/* Categorical palette — fixed order, never cycled (design-system CHART_COLORS). */
export const CHART_COLORS = [
  "var(--o1-chart-1)",
  "var(--o1-chart-2)",
  "var(--o1-chart-3)",
  "var(--o1-chart-4)",
  "var(--o1-chart-5)",
  "var(--o1-chart-6)",
];

export function fmtCompact(n: number): string {
  const trim = (v: number) => `${parseFloat(v.toFixed(1))}`;
  if (Math.abs(n) >= 1_000_000) return `${trim(n / 1_000_000)}M`;
  if (Math.abs(n) >= 1_000) return `${trim(n / 1_000)}K`;
  return `${n}`;
}

/** Round the axis up to a tick step of 1/2/2.5/5 × 10^k with ~`count` ticks. */
function niceScale(max: number, count = 4): { yMax: number; ticks: number[] } {
  const rawStep = Math.max(1e-9, max / count);
  const mag = 10 ** Math.floor(Math.log10(rawStep));
  let step = 10 * mag;
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * mag >= rawStep) {
      step = m * mag;
      break;
    }
  }
  const yMax = step * Math.ceil(max / step);
  const ticks: number[] = [];
  for (let v = 0; v <= yMax + 1e-9; v += step) ticks.push(v);
  return { yMax, ticks };
}

/* ---- Legend — swatch carries identity, text wears ink ---- */
export function Legend({
  items,
}: {
  items: { name: string; color: string }[];
}) {
  return (
    <div className="o1-legend">
      {items.map((it) => (
        <span key={it.name} className="o1-legend__item">
          <span className="o1-legend__swatch" style={{ background: it.color }} />
          {it.name}
        </span>
      ))}
    </div>
  );
}

function ChartFrame({
  title,
  subtitle,
  legend,
  children,
}: {
  title: string;
  subtitle?: string;
  legend?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="o1-chart">
      <div className="o1-chart__head">
        <div>
          <div className="o1-chart__title">{title}</div>
          {subtitle && <div className="o1-chart__subtitle">{subtitle}</div>}
        </div>
        {legend}
      </div>
      {children}
    </div>
  );
}

interface TooltipState {
  index: number;
  px: number;
  py: number;
  pw: number;
}

function useHover(labelCount: number) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<TooltipState | null>(null);

  const onMove = (
    e: React.PointerEvent,
    padL: number,
    padR: number,
    viewWidth = W,
  ) => {
    const el = plotRef.current;
    if (!el || labelCount < 1) return;
    const rect = el.getBoundingClientRect();
    const fracPad = {
      l: (padL / viewWidth) * rect.width,
      r: (padR / viewWidth) * rect.width,
    };
    const usable = rect.width - fracPad.l - fracPad.r;
    const x = e.clientX - rect.left - fracPad.l;
    const index = Math.min(
      labelCount - 1,
      Math.max(0, Math.round((x / usable) * (labelCount - 1))),
    );
    setHover({
      index,
      px: e.clientX - rect.left,
      py: e.clientY - rect.top,
      pw: rect.width,
    });
  };

  return { plotRef, hover, setHover, onMove };
}

function Tooltip({
  state,
  title,
  rows,
}: {
  state: TooltipState;
  title: string;
  rows: { name: string; color: string; value: string }[];
}) {
  const flip = state.px > state.pw / 2;
  return (
    <div
      className="o1-tooltip"
      style={{
        left: state.px + 16,
        top: Math.max(state.py - 12, 0),
        transform: flip ? "translateX(calc(-100% - 32px))" : undefined,
      }}
    >
      <div className="o1-tooltip__title">{title}</div>
      {rows.map((r) => (
        <div key={r.name} className="o1-tooltip__row">
          <span className="o1-tooltip__key">
            <span className="o1-legend__swatch" style={{ background: r.color }} />
            {r.name}
          </span>
          <span className="o1-tooltip__val">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   UsageChart — trend over time, one y-axis, crosshair + tooltip
   ============================================================ */
const W = 640;
const H = 280;
const PAD = { t: 14, r: 16, b: 30, l: 48 };

export function UsageChart({
  title,
  subtitle,
  labels,
  series,
}: {
  title: string;
  subtitle?: string;
  labels: string[];
  series: { name: string; points: number[] }[];
}) {
  const { plotRef, hover, setHover, onMove } = useHover(labels.length);

  const { yMax, ticks } = useMemo(
    () => niceScale(Math.max(...series.flatMap((s) => s.points))),
    [series],
  );

  const px = (i: number) =>
    PAD.l + (i / Math.max(1, labels.length - 1)) * (W - PAD.l - PAD.r);
  const py = (v: number) => PAD.t + (1 - v / yMax) * (H - PAD.t - PAD.b);

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={
        <Legend
          items={series.map((s, i) => ({ name: s.name, color: CHART_COLORS[i] }))}
        />
      }
    >
      <div
        className="o1-chart__plot"
        ref={plotRef}
        onPointerMove={(e) => onMove(e, PAD.l, PAD.r)}
        onPointerLeave={() => setHover(null)}
      >
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
          {ticks.map((t) => (
            <g key={t}>
              <line
                className={t === 0 ? "o1-chart__baseline" : "o1-chart__gridline"}
                x1={PAD.l}
                x2={W - PAD.r}
                y1={py(t)}
                y2={py(t)}
              />
              <text
                className="o1-chart__tick"
                x={PAD.l - 8}
                y={py(t) + 4}
                textAnchor="end"
              >
                {fmtCompact(t)}
              </text>
            </g>
          ))}
          {labels.map((d, i) => (
            <text
              key={d}
              className="o1-chart__tick"
              x={px(i)}
              y={H - 8}
              textAnchor={i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle"}
            >
              {d}
            </text>
          ))}
          {hover && (
            <line
              className="o1-chart__crosshair"
              x1={px(hover.index)}
              x2={px(hover.index)}
              y1={PAD.t}
              y2={H - PAD.b}
            />
          )}
          {series.map((s, si) => (
            <polyline
              key={s.name}
              points={s.points.map((v, i) => `${px(i)},${py(v)}`).join(" ")}
              fill="none"
              stroke={CHART_COLORS[si]}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {hover &&
            series.map((s, si) => (
              <circle
                key={s.name}
                cx={px(hover.index)}
                cy={py(s.points[hover.index])}
                r="4.5"
                fill={CHART_COLORS[si]}
                stroke="var(--o1-surface)"
                strokeWidth="2"
              />
            ))}
        </svg>
        {hover && (
          <Tooltip
            state={hover}
            title={labels[hover.index]}
            rows={series.map((s, si) => ({
              name: s.name,
              color: CHART_COLORS[si],
              value: s.points[hover.index].toLocaleString(),
            }))}
          />
        )}
      </div>
    </ChartFrame>
  );
}

/* ============================================================
   TokenBars — input vs output as stacked columns (≤24px wide,
   2px surface gap between segments, rounded top data-end)
   ============================================================ */
const INPUT_COLOR = "var(--o1-cyan-300)";
const OUTPUT_COLOR = "var(--o1-cyan-600)";

function topRoundedRect(x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, h, w / 2);
  return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h} Z`;
}

export function TokenBars({
  title,
  subtitle,
  labels,
  input,
  output,
}: {
  title: string;
  subtitle?: string;
  labels: string[];
  input: number[];
  output: number[];
}) {
  const { plotRef, hover, setHover, onMove } = useHover(labels.length);

  // narrower viewBox than the line chart — this card sits in the 2fr column,
  // so a ~1:1 scale keeps axis text the same optical size across charts
  const BW = 420;
  const totals = labels.map((_, i) => input[i] + output[i]);
  const { yMax, ticks } = niceScale(Math.max(...totals), 3);

  const step = (BW - PAD.l - PAD.r) / labels.length;
  const barW = Math.min(24, step * 0.55);
  const px = (i: number) => PAD.l + (i + 0.5) * step - barW / 2;
  const py = (v: number) => PAD.t + (1 - v / yMax) * (H - PAD.t - PAD.b);
  const barH = (v: number) => (v / yMax) * (H - PAD.t - PAD.b);

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={
        <Legend
          items={[
            { name: "Input", color: "#67e0f2" },
            { name: "Output", color: "#0891b2" },
          ]}
        />
      }
    >
      <div
        className="o1-chart__plot"
        ref={plotRef}
        onPointerMove={(e) => onMove(e, PAD.l, PAD.r, BW)}
        onPointerLeave={() => setHover(null)}
      >
        <svg viewBox={`0 0 ${BW} ${H}`} role="img" aria-label={title}>
          {ticks.map((t) => (
            <g key={t}>
              <line
                className={t === 0 ? "o1-chart__baseline" : "o1-chart__gridline"}
                x1={PAD.l}
                x2={BW - PAD.r}
                y1={py(t)}
                y2={py(t)}
              />
              <text
                className="o1-chart__tick"
                x={PAD.l - 8}
                y={py(t) + 4}
                textAnchor="end"
              >
                {fmtCompact(t)}
              </text>
            </g>
          ))}
          {labels.map((d, i) => (
            <text
              key={d}
              className="o1-chart__tick"
              x={px(i) + barW / 2}
              y={H - 8}
              textAnchor="middle"
            >
              {d}
            </text>
          ))}
          {labels.map((d, i) => {
            const dimmed = hover !== null && hover.index !== i;
            return (
              <g key={d} opacity={dimmed ? 0.45 : 1}>
                {/* input — bottom segment, anchored to the baseline */}
                <rect
                  x={px(i)}
                  y={py(input[i])}
                  width={barW}
                  height={barH(input[i])}
                  fill={INPUT_COLOR}
                />
                {/* output — top segment with a 2px surface gap + rounded data-end */}
                <path
                  d={topRoundedRect(
                    px(i),
                    py(totals[i]) - 2,
                    barW,
                    barH(output[i]),
                    4,
                  )}
                  fill={OUTPUT_COLOR}
                />
              </g>
            );
          })}
        </svg>
        {hover && (
          <Tooltip
            state={hover}
            title={labels[hover.index]}
            rows={[
              {
                name: "Output",
                color: "#0891b2",
                value: output[hover.index].toLocaleString(),
              },
              {
                name: "Input",
                color: "#67e0f2",
                value: input[hover.index].toLocaleString(),
              },
              {
                name: "Total",
                color: "transparent",
                value: totals[hover.index].toLocaleString(),
              },
            ]}
          />
        )}
      </div>
    </ChartFrame>
  );
}

/* ============================================================
   StatCard — KPI tile with sparkline
   ============================================================ */
export function StatCard({
  label,
  value,
  delta,
  trend,
}: {
  label: string;
  value: string;
  delta?: StatDelta;
  trend?: number[];
}) {
  const spark = useMemo(() => {
    if (!trend || trend.length < 2) return null;
    const min = Math.min(...trend);
    const max = Math.max(...trend);
    const span = max - min || 1;
    const sw = 120;
    const sh = 32;
    const pts = trend.map(
      (v, i) =>
        `${(i / (trend.length - 1)) * sw},${3 + (1 - (v - min) / span) * (sh - 6)}`,
    );
    return { pts: pts.join(" "), last: pts[pts.length - 1].split(",").map(Number), sw, sh };
  }, [trend]);

  const tone =
    delta?.direction === "flat"
      ? "o1-stat__delta--flat"
      : delta?.good
        ? "o1-stat__delta--good"
        : "o1-stat__delta--bad";
  const arrow =
    delta?.direction === "up" ? "↑" : delta?.direction === "down" ? "↓" : "→";

  return (
    <Card flush>
      <div className="o1-stat">
        <span className="o1-stat__label">{label}</span>
        <div className="o1-stat__row">
          <span className="o1-stat__value">{value}</span>
          {delta && (
            <span className={`o1-stat__delta ${tone}`}>
              <span aria-hidden>{arrow}</span> {delta.text}
            </span>
          )}
        </div>
        {spark && (
          <svg
            className="o1-stat__spark"
            viewBox={`0 0 ${spark.sw} ${spark.sh}`}
            preserveAspectRatio="none"
            style={{ height: 32 }}
            aria-hidden
          >
            <polyline
              className="o1-spark-current"
              points={spark.pts}
              fill="none"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={spark.last[0]}
              cy={spark.last[1]}
              r="3"
              fill="var(--o1-cyan-600)"
              stroke="var(--o1-surface)"
              strokeWidth="1.5"
            />
          </svg>
        )}
      </div>
    </Card>
  );
}

/* ============================================================
   Meter — the fill carries severity, always with a text label
   ============================================================ */
export function Meter({
  label,
  ratio,
  valueLabel,
}: {
  label: string;
  ratio: number;
  valueLabel: string;
}) {
  const tone =
    ratio >= 0.95 ? "critical" : ratio >= 0.75 ? "warning" : "accent";
  return (
    <div className={`o1-meter o1-meter--${tone}`}>
      <div className="o1-meter__head">
        <span className="o1-meter__label">{label}</span>
        <span className="o1-meter__value">{valueLabel}</span>
      </div>
      <div className="o1-meter__track">
        <div
          className="o1-meter__fill"
          style={{ width: `${Math.min(100, ratio * 100)}%` }}
        />
      </div>
    </div>
  );
}
