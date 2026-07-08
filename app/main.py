from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat_history import router as chat_history_router
from app.config import get_cors_origins
from app.database import database_status
from app.ui_api import router as ui_router


app = FastAPI(
    title="AMD Hackathon Team 0-1 API",
    summary="Frontend integration API for O(1) chat, history, and usage views.",
    description=(
        "The frontend consumes these endpoints directly. Use `/docs` for the "
        "interactive OpenAPI UI and `/openapi.json` for generated clients."
    ),
    version="0.1.0",
    openapi_tags=[
        {
            "name": "health",
            "description": "Runtime and dependency status used by the top-bar badge.",
        },
        {
            "name": "chat",
            "description": "Auto-routed chat reply contract used by the composer.",
        },
        {
            "name": "chat history",
            "description": "PostgreSQL-backed chat history rail endpoints.",
        },
        {
            "name": "usage",
            "description": "Dashboard metrics derived from saved chat history.",
        },
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ui_router)
app.include_router(chat_history_router)


@app.get(
    "/health",
    tags=["health"],
    summary="Check API and database status",
    description="Returns `database: ok`, `unavailable`, or `not_configured`.",
)
def health() -> dict[str, str]:
    return {"status": "ok", "database": database_status()}


@app.get("/", tags=["health"], summary="API root")
def root() -> dict[str, str]:
    return {"message": "API is running"}
