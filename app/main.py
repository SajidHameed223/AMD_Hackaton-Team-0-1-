from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chat_history import router as chat_history_router
from app.config import get_cors_origins
from app.database import database_status
from app.schemas import ErrorResponse, HealthResponse
from app.ui_api import router as ui_router


API_VERSION = "v1"

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
    expose_headers=["X-Request-ID", "X-API-Version"],
)
app.include_router(ui_router)
app.include_router(chat_history_router)
app.include_router(ui_router, prefix="/api/v1")
app.include_router(chat_history_router, prefix="/api/v1")


@app.middleware("http")
async def add_integration_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-API-Version"] = API_VERSION
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": {
                "code": f"http_{exc.status_code}",
                "message": message,
                "requestId": request_id,
            },
        },
        headers={"X-Request-ID": request_id or "", "X-API-Version": API_VERSION},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "requestId": request_id,
            },
        },
        headers={"X-Request-ID": request_id or "", "X-API-Version": API_VERSION},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check API and database status",
    description="Returns `database: ok`, `unavailable`, or `not_configured`.",
    responses={503: {"model": ErrorResponse}},
)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        database=database_status(),
        apiVersion=API_VERSION,
        requestId=getattr(request.state, "request_id", None),
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check API and database status",
    description="Versioned alias for `GET /health`.",
)
def versioned_health(request: Request) -> HealthResponse:
    return health(request)


@app.get("/", tags=["health"], summary="API root")
def root() -> dict[str, str]:
    return {"message": "API is running", "apiVersion": API_VERSION}


@app.get("/api/v1", tags=["health"], summary="Versioned API root")
def versioned_root() -> dict[str, str]:
    return root()
