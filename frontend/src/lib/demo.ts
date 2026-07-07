import type { Route, UsageSummary } from "./types";

/**
 * Demo mode — a faithful simulation of the O(1) router so the UI is fully
 * demoable before the FastAPI /chat and /usage endpoints land.
 */

export interface DemoReply {
  route: Route;
  model: string;
  latencyMs: number;
  markdown: string;
}

const QUICKSORT = `Here's a compact quicksort. It picks the middle element as the pivot and recurses on the two partitions:

\`\`\`python
def quicksort(xs):
    if len(xs) <= 1:
        return xs
    pivot = xs[len(xs) // 2]
    left  = [x for x in xs if x < pivot]
    mid   = [x for x in xs if x == pivot]
    right = [x for x in xs if x > pivot]
    return quicksort(left) + mid + quicksort(right)

print(quicksort([7, 2, 9, 4, 4, 1]))  # [1, 2, 4, 4, 7, 9]
\`\`\`

Average case is **O(n log n)**; the worst case is O(n²) when the pivot keeps landing on an extreme value. For production code, prefer the built-in \`sorted()\` — it's Timsort, and it's very hard to beat.`;

const BIG_O = `Big-O notation describes how an algorithm's cost grows with input size — it captures the *shape* of the growth curve, not the exact running time. The best case is constant time:

$$
T(n) = O(1) \\iff \\exists\\, c > 0 : T(n) \\le c \\ \\text{ for all } n
$$

A hash-table lookup is the classic example — the work stays flat no matter how many keys you store. Compare that with $O(\\log n)$ for binary search, or $O(n^2)$ for a naive pairwise comparison, where the curve bends up sharply as inputs grow.`;

const OCEAN = `Water absorbs long wavelengths — reds and oranges — much more strongly than short ones, so the sunlight that survives a few meters of seawater is mostly blue, and some of it scatters back up to your eye. Deeper water looks darker and bluer because the light travels farther and loses even more red. Near the coast, sediment and phytoplankton shift the color toward green.`;

const STANDUP = `**Yesterday** — shipped the router fallback path and closed two latency regressions.

**Today** — wiring the usage dashboard to live token counts, then reviewing Ana's caching PR.

**Blockers** — none, though the staging quota resets at noon, so the load test runs after lunch.`;

const DEFAULT_LOCAL = `The router scored this prompt as low-complexity, so it stayed on the local model — sub-100 ms round trip, zero marginal cost. Ask something that needs deeper reasoning (code, math, multi-step planning) and you'll see the turn escalate to the cloud model instead.`;

const DEFAULT_CLOUD = `This one scored above the complexity threshold, so the router escalated it to the cloud model — a longer round trip, but the larger model handles multi-step reasoning far better. You can watch the local/cloud split build up on the Dashboard tab.`;

export function pickDemoReply(text: string): DemoReply {
  const t = text.toLowerCase();
  if (/big[- ]?o|complexit|notation/.test(t)) {
    return { route: "cloud", model: "qwen-72b", latencyMs: 840, markdown: BIG_O };
  }
  if (/quicksort|sort|python|code|function|script/.test(t)) {
    return { route: "cloud", model: "qwen-72b", latencyMs: 1120, markdown: QUICKSORT };
  }
  if (/ocean|blue|water|sea/.test(t)) {
    return { route: "local", model: "llama-3.1-8b", latencyMs: 95, markdown: OCEAN };
  }
  if (/standup|status update|draft/.test(t)) {
    return { route: "local", model: "llama-3.1-8b", latencyMs: 110, markdown: STANDUP };
  }
  return text.length > 90
    ? { route: "cloud", model: "qwen-72b", latencyMs: 910, markdown: DEFAULT_CLOUD }
    : { route: "local", model: "llama-3.1-8b", latencyMs: 92, markdown: DEFAULT_LOCAL };
}

export const DEMO_SUGGESTIONS = [
  "Explain big-O notation",
  "Write quicksort in Python",
  "Why is the ocean blue?",
  "Draft a standup update",
];

/* Model/series order is fixed and identical across charts and the table:
   slot 1 = llama (local, cyan), slot 2 = qwen (cloud, coral). */
export const DEMO_USAGE: UsageSummary = {
  rangeLabel: "Jul 1 – Jul 7, 2026 · all models",
  days: ["Jul 1", "Jul 2", "Jul 3", "Jul 4", "Jul 5", "Jul 6", "Jul 7"],
  routeSeries: [
    { name: "llama-3.1-8b", points: [38400, 42100, 44900, 50200, 52800, 55600, 58900] },
    { name: "qwen-72b", points: [16800, 19400, 21200, 24600, 27100, 30300, 32800] },
  ],
  tokenInput: [34100, 38000, 40900, 46200, 49400, 53100, 56700],
  tokenOutput: [21100, 23500, 25200, 28600, 30500, 32800, 35000],
  modelRows: [
    { model: "llama-3.1-8b", provider: "local", requests: 1284, inputTokens: 214600, outputTokens: 128300, cost: 0 },
    { model: "qwen-72b", provider: "cloud", requests: 412, inputTokens: 103800, outputTokens: 68400, cost: 7.42 },
  ],
  stats: {
    requests: {
      value: "1,696",
      delta: { text: "+12% vs last week", direction: "up", good: true },
      trend: [182, 190, 176, 205, 214, 198, 226, 231, 240, 236, 252, 261],
    },
    tokens: {
      value: "515.1K",
      delta: { text: "+9% vs last week", direction: "up", good: true },
      trend: [55.2, 61.5, 58.0, 66.1, 70.4, 68.2, 74.8, 79.9, 77.1, 85.9, 88.4, 91.7],
    },
    localRate: {
      value: "76%",
      delta: { text: "+4 pts vs last week", direction: "up", good: true },
      trend: [68, 69, 71, 70, 72, 73, 72, 74, 75, 74, 76, 76],
    },
    cost: {
      value: "$7.42",
      delta: { text: "−18% vs last week", direction: "down", good: true },
      trend: [1.62, 1.48, 1.51, 1.3, 1.24, 1.18, 1.02, 1.08, 0.96, 0.9, 0.86, 0.79],
    },
  },
  budget: { spent: 7.42, limit: 25, paceLabel: "$9.80" },
};
