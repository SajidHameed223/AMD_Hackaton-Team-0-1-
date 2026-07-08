# Project Context for Future Agents

This repo is the O(1) hackathon project for Track 1, Agentic AI. The product is
a chat UI plus backend routing layer that sends prompts to different model
backends to balance speed, cost, and response quality.

## Current Team Context

- Science_AJ is the frontend owner. That means React, Next.js, UI polish,
  LaTeX rendering, code snippets, copy buttons, retry button, loading/streaming
  text, and message formatting.
- Hero owns the FastAPI integration layer that connects the frontend to the
  backend services.
- CringeKid owns local LLM work, including Gemma and other local models on WSL.
- Unknown Person owns cloud LLM work, currently involving Fireworks.ai.
- Sajid Hameed and Jae own router logic.
- Sajid is also training or benchmarking router behavior on LeetCode-style
  datasets.

Do not assume Science_AJ is asking for backend model implementation unless the
request explicitly says so. Science_AJ usually needs clean frontend-facing
FastAPI contracts and UI behavior.

## Important Integration Rule

This project is not tied to OpenAI. The model backend can be any local model,
Fireworks.ai model, custom router, or other provider.

FastAPI should expose stable endpoints for the frontend. Backend/model teams can
plug their implementation behind those endpoints. The current generic chat
adapter is:

- `POST /chat`
- Optional forwarding via `CHAT_BACKEND_URL`
- Response shape: `reply`, `route`, `model`, `latency_ms`

The `model` field is only a display string returned by the backend. It must not
be treated as a fixed provider or fixed model name.

## Current API Contract

Use `docs/frontend-api-contract.md` as the source of truth for frontend/backend
integration payloads. FastAPI also exposes:

- `GET /docs`
- `GET /openapi.json`

Current frontend-facing endpoints:

- `GET /health`
- `POST /chat`
- `POST /chat/retry`
- `GET /chat/sessions`
- `GET /chat/sessions/search?q={query}`
- `GET /chat/sessions/{session_id}`
- `PUT /chat/sessions/{session_id}`
- `GET /dashboard/usage`
- `GET /ui/config`

Versioned aliases also exist under `/api/v1` for integration clients. Existing
frontend calls can stay unversioned.

## Database Context

Chat history is PostgreSQL-backed and migratable with Alembic. The repo ships
schema only, with no seed data.

Local setup uses:

- Postgres Docker Compose port `55432`
- `DATABASE_URL=postgresql+psycopg2://o1:o1@localhost:55432/o1`
- `python -m alembic upgrade head`

If `DATABASE_URL` is absent:

- `/health` should return `database: "not_configured"`
- chat history endpoints should return `503`
- `/usage` should still return a valid zeroed dashboard payload

## Frontend Behavior Context

- Routing is auto-only in the UI. Do not restore manual local/cloud selectors.
- The composer should show an `Auto` badge, not a route dropdown.
- Completed assistant messages should show both Copy and Retry actions.
- Retry should resend the original user turn through `POST /chat/retry`, not
  duplicate stale assistant messages.
- The history rail should use PostgreSQL-backed endpoints when available and
  fall back to local draft behavior when unavailable.
- LaTeX rendering and code-block rendering are Science_AJ's frontend lane; do
  not add a FastAPI endpoint for them unless explicitly requested.

## Development Notes

- The working branch for this slice is `frontend`.
- Keep backend changes focused on frontend integration contracts unless asked
  otherwise.
- Keep provider-specific names out of generic docs and UI unless the team asks
  to show a concrete configured model name.
- Preserve `X-Request-ID` and `X-API-Version` response headers when modifying
  FastAPI middleware; they make frontend/backend debugging easier.
- `frontend/AGENTS.md` warns that the Next.js version has breaking changes; read
  local Next.js docs before changing framework-specific code.
