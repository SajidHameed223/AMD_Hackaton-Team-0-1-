from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from remote.config import get_settings
from remote.fireworks_client import call_remote, close_client, stream_remote
from remote.schemas import RemoteRequest, RemoteResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(
    title="AMD Hackathon Team 0-1 API",
    version="0.2.0",
    lifespan=lifespan,
)


def verify_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    if settings.internal_api_key is None:
        return
    if x_internal_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-Internal-Key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "API is running"}


@app.post(
    "/remote",
    response_model=RemoteResponse,
    dependencies=[Depends(verify_internal_key)],
)
async def remote(req: RemoteRequest) -> RemoteResponse:
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /remote/stream for streaming responses.",
        )
    try:
        return await call_remote(req)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fireworks call failed: {e}")


@app.post("/remote/stream", dependencies=[Depends(verify_internal_key)])
async def remote_stream(req: RemoteRequest):
    async def event_generator():
        try:
            async for chunk in stream_remote(req):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
