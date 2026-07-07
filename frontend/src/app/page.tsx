"use client";

import { useCallback, useEffect, useState } from "react";
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
import { CopyButton } from "@/components/chat/CodeBlock";
import { Markdown } from "@/components/chat/Markdown";
import { Dashboard } from "@/components/dash/Dashboard";
import { JoyDecorations } from "@/components/JoyDecorations";
import { LoadingScreen } from "@/components/LoadingScreen";
import { checkHealth, sendChat } from "@/lib/api";
import { DEMO_SUGGESTIONS } from "@/lib/demo";
import type { MessageModel, RoutePreference, RouteVerdict } from "@/lib/types";

let seq = 0;
const nextId = (prefix: string) => `${prefix}${++seq}`;

export default function Home() {
  const [view, setView] = useState<"chat" | "dash">("chat");
  const [loadingDone, setLoadingDone] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [messages, setMessages] = useState<MessageModel[]>([]);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [lastVerdict, setLastVerdict] = useState<RouteVerdict | null>(null);
  const [routePreference, setRoutePreference] =
    useState<RoutePreference>("auto");

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

  const patchMessage = useCallback(
    (
      id: string,
      patch:
        | Partial<MessageModel>
        | ((m: MessageModel) => Partial<MessageModel>),
    ) => {
      setMessages((ms) =>
        ms.map((m) =>
          m.id === id
            ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) }
            : m,
        ),
      );
    },
    [],
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
      setMessages((ms) => [
        ...ms,
        { id: userId, role: "user", content: text, status: "done" },
        { id: asstId, role: "assistant", content: "", status: "routing" },
      ]);
      setPendingId(asstId);

      void sendChat(text, history, routePreference, {
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
    [messages, pendingId, patchMessage, routePreference],
  );

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
        {view === "chat" ? (
          <div className="app-pane">
            <ChatWindow
              scrollRef={scrollRef}
              composer={
                <ChatInput
                  onSend={send}
                  disabled={!!pendingId}
                  placeholder="Ask O(1) anything…"
                  footStart={footBadge}
                  routePreference={routePreference}
                  onRoutePreferenceChange={setRoutePreference}
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
                        <CopyButton text={m.content} surface="light" />
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
      </AppShell>
      <LoadingScreen done={loadingDone} />
    </O1Provider>
  );
}
