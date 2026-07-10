import { DEMO_USAGE, pickDemoReply } from "./demo";
import type {
  ChatStreamEvent,
  ChatSessionRecord,
  ChatSessionSummary,
  ChatTurn,
  MessageModel,
  Route,
  UsageSummary,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "";

export interface ChatCallbacks {
  onRoute: (route: Route, model: string) => void;
  onDelta: (text: string) => void;
  onDone: (route: Route, model: string, latencyMs: number) => void;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return (await res.json()) as T;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(2500),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchChatSessionSummaries(query = ""): Promise<{
  sessions: ChatSessionSummary[];
  live: boolean;
}> {
  try {
    const q = query.trim();
    const path = q
      ? `/chat/sessions/search?q=${encodeURIComponent(q)}`
      : "/chat/sessions";
    const data = await fetchJson<{ sessions: ChatSessionSummary[] }>(
      path,
      { signal: AbortSignal.timeout(3000) },
    );
    return { sessions: data.sessions, live: true };
  } catch {
    return { sessions: [], live: false };
  }
}

export async function fetchChatSession(
  sessionId: string,
): Promise<ChatSessionRecord | null> {
  try {
    return await fetchJson<ChatSessionRecord>(`/chat/sessions/${sessionId}`, {
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    return null;
  }
}

export async function saveChatSession(session: {
  id: string;
  title: string;
  preview: string;
  messages: MessageModel[];
}): Promise<boolean> {
  try {
    await fetchJson<ChatSessionRecord>(`/chat/sessions/${session.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: session.title,
        preview: session.preview,
        messages: session.messages,
      }),
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Send a chat turn. Prefers the FastAPI backend; when /chat is missing or
 * unreachable, falls back to the demo router simulation so the app keeps
 * working end to end.
 */
export async function sendChat(
  message: string,
  history: ChatTurn[],
  cb: ChatCallbacks,
  path = "/chat",
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream, application/json",
      },
      body: JSON.stringify({ message, history }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("text/event-stream") && res.body) {
      await consumeSse(res.body, cb);
      return;
    }

    // Plain JSON reply
    const data = (await res.json()) as {
      reply?: string;
      response?: string;
      route?: Route;
      model?: string;
      latency_ms?: number;
    };
    const reply = data.reply ?? data.response ?? "";
    const route = data.route ?? "local";
    const model = data.model ?? "unknown";
    cb.onRoute(route, model);
    cb.onDelta(reply);
    cb.onDone(route, model, data.latency_ms ?? 0);
  } catch {
    await demoChat(message, cb);
  }
}

async function consumeSse(
  body: ReadableStream<Uint8Array>,
  cb: ChatCallbacks,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        let event: ChatStreamEvent;
        try {
          event = JSON.parse(payload) as ChatStreamEvent;
        } catch {
          continue;
        }
        if (event.type === "route") cb.onRoute(event.route, event.model);
        else if (event.type === "delta") cb.onDelta(event.text);
        else if (event.type === "done")
          cb.onDone(event.route, event.model, event.latency_ms);
      }
    }
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function demoChat(
  message: string,
  cb: ChatCallbacks,
): Promise<void> {
  const reply = pickDemoReply(message);
  await sleep(700 + Math.random() * 450);
  cb.onRoute(reply.route, reply.model);
  // stream in word-ish chunks, faster for the "fast local" feel
  const step = reply.route === "local" ? 14 : 8;
  for (let i = 0; i < reply.markdown.length; i += step) {
    cb.onDelta(reply.markdown.slice(i, i + step));
    await sleep(16);
  }
  cb.onDone(reply.route, reply.model, reply.latencyMs);
}

export async function fetchUsage(): Promise<{
  usage: UsageSummary;
  live: boolean;
}> {
  try {
    const usage = await fetchJson<UsageSummary>("/dashboard/usage", {
      signal: AbortSignal.timeout(3000),
    });
    return { usage, live: true };
  } catch {
    return { usage: DEMO_USAGE, live: false };
  }
}
