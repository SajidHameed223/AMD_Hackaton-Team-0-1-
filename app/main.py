from fastapi import FastAPI

from app.chat_history import router as chat_history_router
from app.database import database_status
from app.ui_api import router as ui_router


app = FastAPI(title="AMD Hackathon Team 0-1 API", version="0.1.0")
app.include_router(ui_router)
app.include_router(chat_history_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": database_status()}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "API is running"}
