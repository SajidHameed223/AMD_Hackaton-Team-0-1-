/**
 * O(1) frontend ↔ FastAPI contract.
 *
 * Backend endpoints the frontend consumes (see frontend/README.md):
 *   GET  /health  -> { status: "ok" }
 *   POST /chat    -> SSE stream of ChatStreamEvent, or a plain ChatResponse JSON
 *   GET  /usage   -> UsageSummary
 *
 * Until an endpoint exists, the frontend silently runs in demo mode.
 */

export type Route = "local" | "cloud";

export interface RouteVerdict {
  route: Route;
  model: string;
  latencyMs?: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatTurn[];
}

/** Non-streaming reply shape. */
export interface ChatResponse {
  reply: string;
  route: Route;
  model: string;
  latency_ms: number;
}

/** Server-sent events emitted by POST /chat when streaming. */
export type ChatStreamEvent =
  | { type: "route"; route: Route; model: string }
  | { type: "delta"; text: string }
  | { type: "done"; route: Route; model: string; latency_ms: number };

export interface StatDelta {
  text: string;
  direction: "up" | "down" | "flat";
  good: boolean;
}

export interface ModelRow {
  model: string;
  provider: Route;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
}

export interface UsageSummary {
  rangeLabel: string;
  days: string[];
  routeSeries: { name: string; points: number[] }[];
  tokenInput: number[];
  tokenOutput: number[];
  modelRows: ModelRow[];
  stats: {
    requests: { value: string; delta: StatDelta; trend: number[] };
    tokens: { value: string; delta: StatDelta; trend: number[] };
    localRate: { value: string; delta: StatDelta; trend: number[] };
    cost: { value: string; delta: StatDelta; trend: number[] };
  };
  budget: { spent: number; limit: number; paceLabel: string };
}

export interface MessageModel {
  id: string;
  role: "user" | "assistant";
  /** Markdown source of the turn. */
  content: string;
  status: "routing" | "streaming" | "done";
  verdict?: RouteVerdict;
}
