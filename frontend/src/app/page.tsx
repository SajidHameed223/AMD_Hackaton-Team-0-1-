"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell, Badge, O1Provider, Tabs, TopBar, Logo } from "@/components/core";
import {
  ChatInput,
  ChatWelcome,
  ChatWindow,
  Message,
  RouteBadge,
  StreamingText,
  TypingIndicator,
  useAutoScroll,
} from "@/components/chat/chat";
import { CopyButton, RetryButton } from "@/components/chat/CodeBlock";
import { Markdown } from "@/components/chat/Markdown";
import { Dashboard } from "@/components/dash/Dashboard";
import { JoyDecorations } from "@/components/JoyDecorations";
import { LoadingScreen } from "@/components/LoadingScreen";
import {
  checkHealth,
  fetchChatSession,
  fetchChatSessionSummaries,
  saveChatSession,
  sendChat,
} from "@/lib/api";
import { DEMO_SUGGESTIONS } from "@/lib/demo";
import type {
  ChatSessionRecord,
  ChatSessionSummary,
  MessageModel,
  RouteVerdict,
} from "@/lib/types";

let seq = 0;
const nextId = (prefix: string) => `${prefix}${++seq}`;

interface ChatSession {
  id: string;
  title: string;
  preview: string;
  messages: MessageModel[];
  updatedAt: number;
  messageCount: number;
}

const makeSessionId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
    (
      Number(c) ^
      (Math.random() * 16) >>
        (Number(c) / 4)
    ).toString(16),
  );
};

const emptySession = (): ChatSession => ({
  id: makeSessionId(),
  title: "New chat",
  preview: "Ask O(1) anything",
  messages: [],
  updatedAt: Date.now(),
  messageCount: 0,
});

const newSession = (): ChatSession => ({
  ...emptySession(),
  updatedAt: Date.now(),
});

const summarizeSession = (messages: MessageModel[]) => {
  const firstUser = messages.find((m) => m.role === "user")?.content.trim();
  const latest = [...messages].reverse().find((m) => m.content.trim())?.content.trim();
  const clean = (text?: string) => (text ? text.replace(/\s+/g, " ") : "");
  const title = clean(firstUser) || "New chat";
  const preview = clean(latest) || "Ask O(1) anything";
  return {
    title: title.slice(0, 54),
    preview: preview.slice(0, 78),
  };
};

const fromSummary = (session: ChatSessionSummary): ChatSession => ({
  id: session.id,
  title: session.title,
  preview: session.preview,
  messages: [],
  updatedAt: new Date(session.updatedAt).getTime(),
  messageCount: session.messageCount,
});

const fromRecord = (session: ChatSessionRecord): ChatSession => {
  const updatedAt = new Date(session.updatedAt).getTime();
  return {
    id: session.id,
    title: session.title,
    preview: session.preview,
    messages: session.messages,
    updatedAt: Number.isFinite(updatedAt) ? updatedAt : Date.now(),
    messageCount: session.messages.length,
  };
};

const formatSessionTime = (updatedAt: number) => {
  if (!updatedAt) return "Now";
  const age = Date.now() - updatedAt;
  if (age < 60_000) return "Now";
  if (age < 3_600_000) return `${Math.max(1, Math.round(age / 60_000))}m ago`;
  if (age < 86_400_000) return `${Math.round(age / 3_600_000)}h ago`;
  if (age < 604_800_000) return `${Math.round(age / 86_400_000)}d ago`;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(
    new Date(updatedAt),
  );
};

export default function Home() {
  const [view, setView] = useState<"chat" | "dash">("chat");
  const [loadingDone, setLoadingDone] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>(() => [emptySession()]);
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0].id);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyLive, setHistoryLive] = useState<boolean | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [lastVerdict, setLastVerdict] = useState<RouteVerdict | null>(null);

  const activeSession = useMemo(
    () =>
      sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [activeSessionId, sessions],
  );
  const messages = useMemo(
    () => activeSession?.messages ?? [],
    [activeSession],
  );

  // opening scene — let the wordmark take one ride before the app fades in
  useEffect(() => {
    const t = setTimeout(() => setLoadingDone(true), 2400);
    return () => clearTimeout(t);
  }, []);

  // backend heartbeat for the status badge
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      const ok = await checkHealth();
      if (alive) setApiOnline(ok);
    };
    ping();
    const t = setInterval(ping, 30_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    let alive = true;
    const t = window.setTimeout(() => {
      void (async () => {
        const result = await fetchChatSessionSummaries();
        if (!alive) return;
        setHistoryLive(result.live);
        if (!result.live || result.sessions.length === 0) {
          const session = emptySession();
          setSessions([session]);
          setActiveSessionId(session.id);
          return;
        }
        const summaries = result.sessions.map(fromSummary);
        const first = await fetchChatSession(summaries[0].id);
        if (!alive) return;
        const loaded = first ? fromRecord(first) : summaries[0];
        setSessions([loaded, ...summaries.slice(1)]);
        setActiveSessionId(loaded.id);
      })();
    }, 0);
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
  }, []);

  useEffect(() => {
    if (!historyLive || !activeSession || activeSession.messages.length === 0) {
      return;
    }
    const t = window.setTimeout(() => {
      void saveChatSession(activeSession);
    }, 450);
    return () => window.clearTimeout(t);
  }, [activeSession, historyLive]);

  const updateActiveMessages = useCallback(
    (
      updater:
        | MessageModel[]
        | ((messages: MessageModel[]) => MessageModel[]),
    ) => {
      setSessions((current) =>
        current.map((session) => {
          if (session.id !== activeSessionId) return session;
          const nextMessages =
            typeof updater === "function" ? updater(session.messages) : updater;
          return {
            ...session,
            ...summarizeSession(nextMessages),
            messages: nextMessages,
            updatedAt: Date.now(),
            messageCount: nextMessages.length,
          };
        }),
      );
    },
    [activeSessionId],
  );

  const patchMessage = useCallback(
    (
      id: string,
      patch:
        | Partial<MessageModel>
        | ((m: MessageModel) => Partial<MessageModel>),
    ) => {
      updateActiveMessages((ms) =>
        ms.map((m) =>
          m.id === id
            ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) }
            : m,
        ),
      );
    },
    [updateActiveMessages],
  );

  const send = useCallback(
    (text: string) => {
      if (pendingId) return;
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));
      const userId = nextId("u");
      const asstId = nextId("a");
      updateActiveMessages((ms) => [
        ...ms,
        { id: userId, role: "user", content: text, status: "done" },
        { id: asstId, role: "assistant", content: "", status: "routing" },
      ]);
      setPendingId(asstId);

      void sendChat(text, history, {
        onRoute: (route, model) =>
          patchMessage(asstId, {
            status: "streaming",
            verdict: { route, model },
          }),
        onDelta: (delta) =>
          patchMessage(asstId, (m) => ({
            status: "streaming",
            content: m.content + delta,
          })),
        onDone: (route, model, latencyMs) => {
          const verdict = { route, model, latencyMs };
          patchMessage(asstId, { status: "done", verdict });
          setLastVerdict(verdict);
          setPendingId(null);
        },
      });
    },
    [messages, pendingId, patchMessage, updateActiveMessages],
  );

  const retryMessage = useCallback(
    (assistantId: string) => {
      if (pendingId) return;
      const assistantIndex = messages.findIndex((item) => item.id === assistantId);
      if (assistantIndex <= 0) return;
      let userIndex = assistantIndex - 1;
      while (userIndex >= 0 && messages[userIndex].role !== "user") {
        userIndex -= 1;
      }
      const userMessage = messages[userIndex];
      if (!userMessage || userMessage.role !== "user") return;

      const history = messages.slice(0, userIndex).map((m) => ({
        role: m.role,
        content: m.content,
      }));
      const asstId = nextId("a");
      updateActiveMessages(() => [
        ...messages.slice(0, userIndex + 1),
        { id: asstId, role: "assistant", content: "", status: "routing" },
      ]);
      setPendingId(asstId);
      setLastVerdict(null);

      void sendChat(userMessage.content, history, {
        onRoute: (route, model) =>
          patchMessage(asstId, {
            status: "streaming",
            verdict: { route, model },
          }),
        onDelta: (delta) =>
          patchMessage(asstId, (m) => ({
            status: "streaming",
            content: m.content + delta,
          })),
        onDone: (route, model, latencyMs) => {
          const verdict = { route, model, latencyMs };
          patchMessage(asstId, { status: "done", verdict });
          setLastVerdict(verdict);
          setPendingId(null);
        },
      });
    },
    [messages, patchMessage, pendingId, updateActiveMessages],
  );

  const startNewChat = useCallback(() => {
    if (pendingId) return;
    if ((activeSession?.messages.length ?? 0) === 0) {
      setView("chat");
      setHistoryQuery("");
      return;
    }
    const session = newSession();
    setSessions((current) => [session, ...current]);
    setActiveSessionId(session.id);
    setView("chat");
    setLastVerdict(null);
    setHistoryQuery("");
  }, [activeSession, pendingId]);

  const openSession = useCallback(
    (id: string) => {
      if (pendingId) return;
      setActiveSessionId(id);
      setView("chat");
      const session = sessions.find((item) => item.id === id);
      const lastAssistant = [...(session?.messages ?? [])]
        .reverse()
        .find((item) => item.role === "assistant" && item.verdict);
      setLastVerdict(lastAssistant?.verdict ?? null);
      if (historyLive && session && session.messages.length === 0 && session.messageCount > 0) {
        void fetchChatSession(id).then((record) => {
          if (!record) return;
          const loaded = fromRecord(record);
          setSessions((current) =>
            current.map((item) => (item.id === id ? loaded : item)),
          );
          const loadedLastAssistant = [...loaded.messages]
            .reverse()
            .find((item) => item.role === "assistant" && item.verdict);
          setLastVerdict(loadedLastAssistant?.verdict ?? null);
        });
      }
    },
    [historyLive, pendingId, sessions],
  );

  const filteredSessions = useMemo(() => {
    const q = historyQuery.trim().toLowerCase();
    const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!q) return sorted;
    return sorted.filter((session) =>
      `${session.title} ${session.preview}`.toLowerCase().includes(q),
    );
  }, [historyQuery, sessions]);

  const scrollRef = useAutoScroll(messages);

  const pending = pendingId ? messages.find((m) => m.id === pendingId) : null;
  const footBadge = pending ? (
    pending.status === "routing" || !pending.verdict ? (
      <RouteBadge route="routing" />
    ) : (
      <RouteBadge route={pending.verdict.route} model={pending.verdict.model} />
    )
  ) : lastVerdict ? (
    <RouteBadge {...lastVerdict} />
  ) : (
    <RouteBadge route="local" model="llama-3.1-8b" />
  );

  return (
    <O1Provider className="app-viewport">
      <JoyDecorations />
      <AppShell
        topBar={
          <TopBar
            brand={<Logo />}
            nav={
              <Tabs
                aria-label="View"
                value={view}
                onChange={(id) => setView(id as "chat" | "dash")}
                items={[
                  { id: "chat", label: "Chat" },
                  { id: "dash", label: "Dashboard" },
                ]}
              />
            }
            side={
              apiOnline === null ? (
                <Badge tone="sand" dot>
                  connecting…
                </Badge>
              ) : apiOnline ? (
                <Badge tone="good" dot>
                  2 models online
                </Badge>
              ) : (
                <Badge tone="sun" dot>
                  demo mode
                </Badge>
              )
            }
          />
        }
      >
        <div className="app-workspace">
          <ChatHistorySidebar
            sessions={filteredSessions}
            activeSessionId={activeSessionId}
            query={historyQuery}
            disabled={!!pendingId}
            historyLive={historyLive}
            onQueryChange={setHistoryQuery}
            onNewChat={startNewChat}
            onOpenSession={openSession}
            onBrowseAll={() => setHistoryQuery("")}
            onOpenDashboard={() => setView("dash")}
            dashboardActive={view === "dash"}
          />
          {view === "chat" ? (
            <div className="app-pane">
              <ChatWindow
                key={activeSessionId}
                scrollRef={scrollRef}
                composer={
                  <ChatInput
                    onSend={send}
                    disabled={!!pendingId}
                    placeholder="Ask O(1) anything..."
                    footStart={footBadge}
                  />
                }
              >
                {messages.length === 0 && (
                  <ChatWelcome
                    title={
                      <>
                        Ask anything. Routed in <em>O(1)</em>.
                      </>
                    }
                    subtitle="Small prompts stay on the local model; hard ones ride out to the cloud."
                    suggestions={DEMO_SUGGESTIONS}
                    onPick={send}
                  />
                )}
                {messages.map((m) =>
                  m.role === "user" ? (
                    <Message key={m.id} role="user">
                      {m.content}
                    </Message>
                  ) : (
                    <Message
                      key={m.id}
                      role="assistant"
                      meta={
                        m.status === "routing" || !m.verdict ? (
                          <RouteBadge route="routing" />
                        ) : (
                          <RouteBadge
                            route={m.verdict.route}
                            model={m.verdict.model}
                            latencyMs={
                              m.status === "done"
                                ? m.verdict.latencyMs
                                : undefined
                            }
                          />
                        )
                      }
                      actions={
                        m.status === "done" ? (
                          <>
                            <CopyButton text={m.content} surface="light" />
                            <RetryButton
                              onRetry={() => retryMessage(m.id)}
                              disabled={!!pendingId}
                              surface="light"
                            />
                          </>
                        ) : undefined
                      }
                    >
                      {m.status === "routing" ? (
                        <TypingIndicator />
                      ) : m.status === "streaming" ? (
                        <StreamingText active>
                          <Markdown streaming>{m.content}</Markdown>
                        </StreamingText>
                      ) : (
                        <Markdown>{m.content}</Markdown>
                      )}
                    </Message>
                  ),
                )}
              </ChatWindow>
            </div>
          ) : (
            <div className="app-pane app-pane--scroll">
              <Dashboard />
            </div>
          )}
        </div>
      </AppShell>
      <LoadingScreen done={loadingDone} />
    </O1Provider>
  );
}

function ChatHistorySidebar({
  sessions,
  activeSessionId,
  query,
  disabled,
  dashboardActive,
  historyLive,
  onQueryChange,
  onNewChat,
  onOpenSession,
  onBrowseAll,
  onOpenDashboard,
}: {
  sessions: ChatSession[];
  activeSessionId: string;
  query: string;
  disabled: boolean;
  dashboardActive: boolean;
  historyLive: boolean | null;
  onQueryChange: (query: string) => void;
  onNewChat: () => void;
  onOpenSession: (id: string) => void;
  onBrowseAll: () => void;
  onOpenDashboard: () => void;
}) {
  return (
    <aside className="o1-history" aria-label="Chat history">
      <div className="o1-history__top">
        <button
          type="button"
          className="o1-history__new"
          onClick={onNewChat}
          disabled={disabled}
        >
          <PlusIcon />
          <span>New chat</span>
        </button>
        <label className="o1-history__search">
          <SearchIcon />
          <span className="o1-sr-only">Search chats</span>
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search chats"
          />
        </label>
        <div className="o1-history__quick" aria-label="Browse">
          <button type="button" onClick={onBrowseAll}>
            <LibraryIcon />
            <span>Browse</span>
          </button>
          <button
            type="button"
            className={dashboardActive ? "o1-history__quick-active" : ""}
            onClick={onOpenDashboard}
          >
            <ChartIcon />
            <span>Usage</span>
          </button>
        </div>
        <div className="o1-history__sync">
          <span
            className={`o1-history__sync-dot${
              historyLive ? " o1-history__sync-dot--live" : ""
            }`}
            aria-hidden
          />
          {historyLive === null
            ? "Checking history"
            : historyLive
              ? "Postgres history"
              : "Local draft"}
        </div>
      </div>

      <div className="o1-history__section">
        <div className="o1-history__label">Recent chats</div>
        <div className="o1-history__list">
          {sessions.length === 0 ? (
            <div className="o1-history__empty">No matching chats</div>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                type="button"
                className={`o1-history__item${
                  session.id === activeSessionId && !dashboardActive
                    ? " o1-history__item--active"
                    : ""
                }`}
                onClick={() => onOpenSession(session.id)}
                disabled={disabled}
              >
                <span className="o1-history__item-title">{session.title}</span>
                <span className="o1-history__item-preview">{session.preview}</span>
                <span className="o1-history__item-time">
                  {formatSessionTime(session.updatedAt)}
                </span>
              </button>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M8 3.5v9M3.5 8h9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M7.1 12.2a5.1 5.1 0 1 1 0-10.2 5.1 5.1 0 0 1 0 10.2Zm3.8-1.3 3.1 3.1"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M3 3.2h8.7A1.3 1.3 0 0 1 13 4.5v8.3H4.1A1.1 1.1 0 0 1 3 11.7V3.2Zm2.2 2.6h5.4M5.2 8h4.3"
        stroke="currentColor"
        strokeWidth="1.45"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M3 12.8V3.2m0 9.6h10M5.4 10.2V7.4m3 2.8V4.8m3 5.4V6.2"
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
      />
    </svg>
  );
}
