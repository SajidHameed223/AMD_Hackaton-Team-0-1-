"use client";

import { useEffect, useState } from "react";
import { fetchUsage } from "@/lib/api";
import type { UsageSummary } from "@/lib/types";
import { Badge, Card, Skeleton, WaveDivider } from "../core";
import {
  CHART_COLORS,
  Meter,
  StatCard,
  TokenBars,
  UsageChart,
  fmtCompact,
} from "./charts";

function ModelTable({ usage }: { usage: UsageSummary }) {
  return (
    <table className="o1-table">
      <thead>
        <tr>
          <th>Model</th>
          <th className="o1-table__num">Requests</th>
          <th className="o1-table__num">Input tokens</th>
          <th className="o1-table__num">Output tokens</th>
          <th className="o1-table__num">Cost</th>
        </tr>
      </thead>
      <tbody>
        {usage.modelRows.map((row, i) => (
          <tr key={row.model}>
            <td>
              <span className="o1-table__model">
                <span
                  className="o1-table__swatch"
                  style={{ background: CHART_COLORS[i] }}
                />
                {row.model}
                <span className="o1-table__sub">{row.provider}</span>
              </span>
            </td>
            <td className="o1-table__num">{row.requests.toLocaleString()}</td>
            <td className="o1-table__num">{fmtCompact(row.inputTokens)}</td>
            <td className="o1-table__num">{fmtCompact(row.outputTokens)}</td>
            <td className="o1-table__num">
              {row.cost === 0 ? "free" : `$${row.cost.toFixed(2)}`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function DashSkeleton() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="o1-kpis">
        {[0, 1, 2, 3].map((i) => (
          <Card key={i}>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Skeleton width="55%" />
              <Skeleton width="40%" height={28} />
            </div>
          </Card>
        ))}
      </div>
      <div className="dash-grid-charts">
        <Card>
          <Skeleton height={240} />
        </Card>
        <Card>
          <Skeleton height={240} />
        </Card>
      </div>
    </div>
  );
}

export function Dashboard() {
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // let the skeleton breathe so the reveal doesn't flash
    const start = Date.now();
    fetchUsage().then(({ usage, live }) => {
      const wait = Math.max(0, 650 - (Date.now() - start));
      setTimeout(() => {
        if (cancelled) return;
        setUsage(usage);
        setLive(live);
      }, wait);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 12,
          }}
        >
          <h2
            style={{
              fontFamily: "var(--o1-font-display)",
              fontSize: 22,
              fontWeight: 600,
              color: "var(--o1-ink)",
            }}
          >
            Usage
          </h2>
          {usage && !live && <Badge tone="sand">sample data</Badge>}
        </div>
        <div style={{ fontSize: 13, color: "var(--o1-ink-3)" }}>
          {usage?.rangeLabel ?? ""}
        </div>
      </div>

      {!usage ? (
        <DashSkeleton />
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="o1-kpis">
            <StatCard label="Requests" {...usage.stats.requests} />
            <StatCard label="Total tokens" {...usage.stats.tokens} />
            <StatCard label="Local hit rate" {...usage.stats.localRate} />
            <StatCard label="Cloud cost" {...usage.stats.cost} />
          </div>

          <div className="dash-grid-charts">
            <Card>
              <UsageChart
                title="Tokens by route"
                subtitle="Local vs cloud · last 7 days"
                labels={usage.days}
                series={usage.routeSeries}
              />
            </Card>
            <Card>
              <TokenBars
                title="Input vs output"
                subtitle="Tokens per day"
                labels={usage.days}
                input={usage.tokenInput}
                output={usage.tokenOutput}
              />
            </Card>
          </div>

          <WaveDivider />

          <div className="dash-grid-bottom">
            <Card flush>
              <ModelTable usage={usage} />
            </Card>
            <Card>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <Meter
                  label="Monthly budget"
                  ratio={usage.budget.spent / usage.budget.limit}
                  valueLabel={`$${usage.budget.spent.toFixed(2)} of $${usage.budget.limit.toFixed(2)}`}
                />
                <div
                  style={{
                    fontSize: 13,
                    lineHeight: 1.55,
                    color: "var(--o1-ink-2)",
                  }}
                >
                  {usage.stats.localRate.value} of turns stayed on the local
                  model at zero marginal cost. Cloud spend is on pace for{" "}
                  <strong>{usage.budget.paceLabel}</strong> this month.
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
