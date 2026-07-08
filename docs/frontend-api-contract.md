# Frontend API Contract

This is the stable FastAPI contract for the O(1) frontend. The generated
FastAPI reference is also available from a running backend at:

- `GET /docs`
- `GET /openapi.json`

## Environment

Backend:

```bash
DATABASE_URL=postgresql+psycopg2://o1:o1@localhost:55432/o1
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CHAT_BACKEND_URL=http://localhost:9000/chat
LOCAL_MODEL_NAME=local-model
CLOUD_MODEL_NAME=cloud-model
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

## Integration Conventions

- Current frontend paths are unversioned for compatibility.
- Versioned aliases also exist under `/api/v1` for backend integrators.
- Every response includes:
  - `X-Request-ID`
  - `X-API-Version`
- Clients can send `X-Request-ID`; otherwise FastAPI generates one.
- Browser clients can read those headers because CORS exposes them.
- Error responses keep FastAPI's familiar `detail` field and also include an
  `error` object:

```json
{
  "detail": "Chat session not found",
  "error": {
    "code": "http_404",
    "message": "Chat session not found",
    "requestId": "5f5e0ef4-0e4d-4cf0-9a10-85790fd6d2d6"
  }
}
```

Versioned aliases:

- `GET /api/v1/health`
- `POST /api/v1/chat`
- `POST /api/v1/chat/retry`
- `GET /api/v1/chat/sessions`
- `GET /api/v1/chat/sessions/search?q={query}`
- `GET /api/v1/chat/sessions/{session_id}`
- `PUT /api/v1/chat/sessions/{session_id}`
- `GET /api/v1/usage`
- `GET /api/v1/dashboard/usage`
- `GET /api/v1/ui/config`

## UI Function to Endpoint Map

This table is the integration checklist by UI function.

| UI function | Method | Path | Versioned path | Backend needed? |
| --- | --- | --- | --- | --- |
| Top-bar API/database status | `GET` | `/health` | `/api/v1/health` | Yes |
| Composer Send | `POST` | `/chat` | `/api/v1/chat` | Yes |
| Assistant Retry button | `POST` | `/chat/retry` | `/api/v1/chat/retry` | Yes |
| History Browse button | `GET` | `/chat/sessions` | `/api/v1/chat/sessions` | Yes |
| History Search input | `GET` | `/chat/sessions/search?q={query}` | `/api/v1/chat/sessions/search?q={query}` | Yes |
| Open old chat | `GET` | `/chat/sessions/{session_id}` | `/api/v1/chat/sessions/{session_id}` | Yes |
| Persist active chat | `PUT` | `/chat/sessions/{session_id}` | `/api/v1/chat/sessions/{session_id}` | Yes |
| Dashboard Usage tab | `GET` | `/dashboard/usage` | `/api/v1/dashboard/usage` | Yes |
| Machine-readable UI map | `GET` | `/ui/config` | `/api/v1/ui/config` | Yes |
| Copy assistant response | `CLIENT` | browser clipboard | browser clipboard | No |
| Copy code block | `CLIENT` | browser clipboard | browser clipboard | No |
| New chat empty draft | `CLIENT` | local draft until first save | local draft until first save | No |
| LaTeX/code rendering | `CLIENT` | Markdown renderer | Markdown renderer | No |

`GET /ui/config` returns this same mapping as JSON so integration clients can
discover the current paths programmatically.

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
  "database": "ok",
  "apiVersion": "v1",
  "requestId": "5f5e0ef4-0e4d-4cf0-9a10-85790fd6d2d6"
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
  "model": "cloud-model",
  "latency_ms": 840
}
```

Notes:

- Routing is auto-only in the UI. Do not send a manual route preference.
- The retry button calls this same endpoint with the original user message and
  prior history before that turn.
- `route` is always `local` or `cloud`.
- `model` is any backend-provided display name. It is not tied to any provider.
- If `CHAT_BACKEND_URL` is set, FastAPI forwards this payload to that backend.
  The backend can use any model or router as long as it returns the response
  shape above.
- If the configured backend is unreachable, FastAPI returns `502`.

## Retry Chat Reply

Purpose: explicit path for the assistant Retry button.

Request:

```http
POST /chat/retry
Content-Type: application/json
```

Body: same shape as `POST /chat`.

Response: same shape as `POST /chat`.

Notes:

- The frontend calls this route for retry so Hero can wire or test retry
  separately from first-send chat traffic.
- `POST /api/v1/chat/retry` is the versioned alias.

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
- `GET /api/v1/chat/sessions` is the versioned alias.

## Search Chat Sessions

Purpose: history search input.

Request:

```http
GET /chat/sessions/search?q=database
```

Response: same shape as `GET /chat/sessions`.

Behavior:

- Searches saved session `title` and `preview`.
- Empty `q` returns recent sessions.
- `GET /api/v1/chat/sessions/search?q={query}` is the versioned alias.

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
        "model": "cloud-model",
        "latencyMs": 840
      }
    }
  ]
}
```

Empty/error behavior:

- Unknown `session_id` returns `404`.
- Missing `DATABASE_URL` returns `503`.
- `GET /api/v1/chat/sessions/{session_id}` is the versioned alias.

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
        "model": "cloud-model",
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
- `PUT /api/v1/chat/sessions/{session_id}` is the versioned alias.

## Usage Dashboard

Purpose: dashboard KPIs, charts, model table, and budget meter.

Request:

```http
GET /dashboard/usage
```

Response shape:

```json
{
  "rangeLabel": "Jul 2 - Jul 8 · Postgres history",
  "days": ["Jul 2", "Jul 3", "Jul 4", "Jul 5", "Jul 6", "Jul 7", "Jul 8"],
  "routeSeries": [{ "name": "cloud-model", "points": [0, 0, 0, 0, 0, 0, 10] }],
  "tokenInput": [0, 0, 0, 0, 0, 0, 5],
  "tokenOutput": [0, 0, 0, 0, 0, 0, 10],
  "modelRows": [
    {
      "model": "cloud-model",
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
- `GET /usage` remains available as a compatibility alias.
- `GET /api/v1/dashboard/usage` is the versioned dashboard path.
- `GET /api/v1/usage` is also available as a compatibility alias.

## UI Config

Purpose: machine-readable endpoint map for frontend/backend integration.

Request:

```http
GET /ui/config
```

Response shape:

```json
{
  "apiVersion": "v1",
  "endpoints": [
    {
      "label": "Dashboard usage",
      "method": "GET",
      "path": "/dashboard/usage",
      "versionedPath": "/api/v1/dashboard/usage",
      "description": "Dashboard KPIs, charts, model table, and budget meter."
    },
    {
      "label": "New chat empty draft",
      "method": "CLIENT",
      "path": "local-draft",
      "versionedPath": "local-draft",
      "description": "Client-only empty session until the first saved message."
    },
    {
      "label": "LaTeX/code rendering",
      "method": "CLIENT",
      "path": "markdown-renderer",
      "versionedPath": "markdown-renderer",
      "description": "Client-only Markdown, LaTeX, and code-block rendering."
    }
  ]
}
```

`GET /api/v1/ui/config` is the versioned alias.

## Frontend Integration Checklist

- Use `NEXT_PUBLIC_API_URL` as the base URL.
- Read `/health` on interval for status.
- Send composer requests to `POST /chat`.
- Send retry requests to `POST /chat/retry`.
- Save the active chat with `PUT /chat/sessions/{session_id}` after messages change.
- Load the rail with `GET /chat/sessions`.
- Search saved chats with `GET /chat/sessions/search?q={query}`.
- Lazy-load full messages with `GET /chat/sessions/{session_id}` when a saved chat is opened.
- Render dashboard from `GET /dashboard/usage` or compatibility path `GET /usage`.
- Treat `503` from chat history as "Local draft" mode.
- Log `X-Request-ID` when reporting backend integration bugs.
