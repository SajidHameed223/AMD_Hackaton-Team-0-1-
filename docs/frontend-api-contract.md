# Frontend API Contract

This is the stable FastAPI contract for the O(1) frontend. The generated API
reference is also available from a running backend at:

- `GET /docs`
- `GET /openapi.json`

## Environment

Backend:

```bash
DATABASE_URL=postgresql+psycopg2://o1:o1@localhost:55432/o1
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

Frontend:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Endpoint Template

Each endpoint below follows this format:

- Purpose: what UI surface uses it.
- Request: method, path, and JSON body when required.
- Response: success payload shape.
- Empty/error behavior: what the frontend should expect when data is absent.

## Health

Purpose: top-bar backend/database status.

Request:

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "database": "ok"
}
```

`database` can be:

- `ok`
- `unavailable`
- `not_configured`

## Chat Reply

Purpose: composer send and retry actions.

Request:

```http
POST /chat
Content-Type: application/json
```

```json
{
  "message": "Explain big-O notation",
  "history": [
    { "role": "user", "content": "Previous user turn" },
    { "role": "assistant", "content": "Previous assistant turn" }
  ]
}
```

Response:

```json
{
  "reply": "Big-O describes how work grows as input grows.",
  "route": "cloud",
  "model": "qwen-72b",
  "latency_ms": 840
}
```

Notes:

- Routing is auto-only in the UI. Do not send a manual route preference.
- The retry button calls this same endpoint with the original user message and
  prior history before that turn.
- `route` is always `local` or `cloud`.

## List Chat Sessions

Purpose: left history rail.

Request:

```http
GET /chat/sessions
```

Response:

```json
{
  "sessions": [
    {
      "id": "55555555-5555-4555-8555-555555555555",
      "title": "Database verification updated",
      "preview": "Second save replaced messages",
      "updatedAt": "2026-07-08T06:47:48.565942Z",
      "messageCount": 2
    }
  ]
}
```

Empty/error behavior:

- Empty database returns `{ "sessions": [] }`.
- Missing `DATABASE_URL` returns `503`.

## Read Chat Session

Purpose: open a saved conversation from the history rail.

Request:

```http
GET /chat/sessions/{session_id}
```

Response:

```json
{
  "id": "55555555-5555-4555-8555-555555555555",
  "title": "Database verification updated",
  "preview": "Second save replaced messages",
  "updatedAt": "2026-07-08T06:47:48.565942Z",
  "messages": [
    {
      "id": "u2",
      "role": "user",
      "content": "Update this session",
      "status": "done",
      "verdict": null
    },
    {
      "id": "a2",
      "role": "assistant",
      "content": "Updated without duplicating old messages.",
      "status": "done",
      "verdict": {
        "route": "cloud",
        "model": "qwen-72b",
        "latencyMs": 840
      }
    }
  ]
}
```

Empty/error behavior:

- Unknown `session_id` returns `404`.
- Missing `DATABASE_URL` returns `503`.

## Save Chat Session

Purpose: persist the current conversation after a turn changes.

Request:

```http
PUT /chat/sessions/{session_id}
Content-Type: application/json
```

```json
{
  "title": "Database verification updated",
  "preview": "Second save replaced messages",
  "messages": [
    {
      "id": "u2",
      "role": "user",
      "content": "Update this session",
      "status": "done"
    },
    {
      "id": "a2",
      "role": "assistant",
      "content": "Updated without duplicating old messages.",
      "status": "done",
      "verdict": {
        "route": "cloud",
        "model": "qwen-72b",
        "latencyMs": 840
      }
    }
  ]
}
```

Response: same shape as `GET /chat/sessions/{session_id}`.

Behavior:

- Creates the session if it does not exist.
- Updates title, preview, and timestamp if it exists.
- Replaces the message list for that session, so retries do not duplicate old
  assistant messages.
- Invalid message roles or statuses return `422`.

## Usage Dashboard

Purpose: dashboard KPIs, charts, model table, and budget meter.

Request:

```http
GET /usage
```

Response shape:

```json
{
  "rangeLabel": "Jul 2 - Jul 8 · Postgres history",
  "days": ["Jul 2", "Jul 3", "Jul 4", "Jul 5", "Jul 6", "Jul 7", "Jul 8"],
  "routeSeries": [{ "name": "qwen-72b", "points": [0, 0, 0, 0, 0, 0, 10] }],
  "tokenInput": [0, 0, 0, 0, 0, 0, 5],
  "tokenOutput": [0, 0, 0, 0, 0, 0, 10],
  "modelRows": [
    {
      "model": "qwen-72b",
      "provider": "cloud",
      "requests": 1,
      "inputTokens": 0,
      "outputTokens": 10,
      "cost": 0.00015
    }
  ],
  "stats": {
    "requests": {
      "value": "1",
      "delta": { "text": "live from Postgres", "direction": "flat", "good": true },
      "trend": [0, 0, 0, 0, 0, 0, 1]
    },
    "tokens": {
      "value": "15",
      "delta": { "text": "estimated from text", "direction": "flat", "good": true },
      "trend": [0, 0, 0, 0, 0, 0, 15]
    },
    "localRate": {
      "value": "0%",
      "delta": { "text": "auto-routing only", "direction": "flat", "good": true },
      "trend": [0, 0, 0, 0, 0, 0, 0]
    },
    "cost": {
      "value": "$0.00",
      "delta": { "text": "estimated cloud spend", "direction": "flat", "good": true },
      "trend": [0, 0, 0, 0, 0, 0, 0.0002]
    }
  },
  "budget": {
    "spent": 0.0,
    "limit": 25.0,
    "paceLabel": "$0.00"
  }
}
```

Empty/error behavior:

- Missing `DATABASE_URL` returns a valid zeroed dashboard payload.
- Empty chat history returns a valid zeroed dashboard payload.

## Frontend Integration Checklist

- Use `NEXT_PUBLIC_API_URL` as the base URL.
- Read `/health` on interval for status.
- Send all composer and retry requests to `POST /chat`.
- Save the active chat with `PUT /chat/sessions/{session_id}` after messages change.
- Load the rail with `GET /chat/sessions`.
- Lazy-load full messages with `GET /chat/sessions/{session_id}` when a saved chat is opened.
- Render dashboard from `GET /usage`.
- Treat `503` from chat history as "Local draft" mode.
