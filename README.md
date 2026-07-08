# AMD Hackathon Team 0-1

Python project bootstrap with local virtual environment and core ML/API dependencies.

## Setup

1. Create virtual environment:

```powershell
py -3 -m venv venv
```

2. Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run once per terminal session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

3. Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Installed Dependencies

- torch==2.12.1
- transformers==5.13.0
- fastapi==0.139.0
- uvicorn==0.50.0
- pydantic==2.13.4
- SQLAlchemy>=2.0,<3.0
- alembic>=1.13,<2.0
- psycopg2-binary>=2.9,<3.0

## Quick Check

```powershell
.\venv\Scripts\python.exe -m pip show torch transformers fastapi uvicorn pydantic
```

## Run API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## PostgreSQL chat history

The API is wired for PostgreSQL-backed chat history, but the repository ships
with no seed data. Each developer can point `DATABASE_URL` at their own local
or hosted Postgres instance, run migrations, and get the same schema.

### Local database

```bash
cp .env.example .env
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg2://o1:o1@localhost:55432/o1
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

On Windows PowerShell, set the variable for the current terminal with:

```powershell
$env:DATABASE_URL="postgresql+psycopg2://o1:o1@localhost:55432/o1"
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

The frontend reads and writes:

- `POST /chat` for one auto-routed assistant reply
- `GET /chat/sessions` for history summaries
- `GET /chat/sessions/{session_id}` for one full conversation
- `PUT /chat/sessions/{session_id}` to upsert a conversation
- `GET /usage` for dashboard metrics shaped from saved chat history

Versioned aliases exist under `/api/v1` for backend integration clients, for
example `POST /api/v1/chat` and `GET /api/v1/chat/sessions`.

For the copy-paste integration contract, see
`docs/frontend-api-contract.md`. FastAPI also exposes live interactive docs at
`/docs` and schema JSON at `/openapi.json`.

If `DATABASE_URL` is missing, `/health` still returns `status: "ok"` with
`database: "not_configured"`, chat history endpoints return `503`, and
`GET /usage` returns a zeroed dashboard payload.

## TODO Status

- [x] Create local virtual environment
- [x] Install project dependencies
- [x] Add requirements.txt
- [x] Add .gitignore for virtual environments
- [x] Add application source code scaffold
- [x] Add PostgreSQL chat-history schema and migrations
