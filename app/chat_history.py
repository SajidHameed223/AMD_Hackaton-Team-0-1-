from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor, execute_batch

from app.database import get_db
from app.schemas import (
    ChatMessageSchema,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatSessionsResponse,
    SaveChatSessionRequest,
)

router = APIRouter(prefix="/chat", tags=["chat history"])


@router.get("/sessions", response_model=ChatSessionsResponse)
def list_sessions(db: PgConnection = Depends(get_db)) -> ChatSessionsResponse:
    with db.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            select
                s.id,
                s.title,
                s.preview,
                s.updated_at as "updatedAt",
                count(m.id)::int as "messageCount"
            from chat_sessions s
            left join chat_messages m on m.session_id = s.id
            group by s.id
            order by s.updated_at desc
            limit 50
            """
        )
        rows = cursor.fetchall()
    return ChatSessionsResponse(
        sessions=[ChatSessionSummary(**row) for row in rows]
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(session_id: UUID, db: PgConnection = Depends(get_db)) -> ChatSessionDetail:
    return _load_session(db, session_id)


@router.put("/sessions/{session_id}", response_model=ChatSessionDetail)
def save_session(
    session_id: UUID,
    payload: SaveChatSessionRequest,
    db: PgConnection = Depends(get_db),
) -> ChatSessionDetail:
    now = datetime.now(timezone.utc)
    title = payload.title or "New chat"
    preview = payload.preview or "Ask O(1) anything"

    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                insert into chat_sessions (id, title, preview, created_at, updated_at)
                values (%s, %s, %s, %s, %s)
                on conflict (id) do update set
                    title = excluded.title,
                    preview = excluded.preview,
                    updated_at = excluded.updated_at
                """,
                (session_id, title, preview, now, now),
            )
            cursor.execute("delete from chat_messages where session_id = %s", (session_id,))
            execute_batch(
                cursor,
                """
                insert into chat_messages (
                    id,
                    session_id,
                    position,
                    role,
                    content,
                    status,
                    route,
                    model,
                    latency_ms
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        message.id,
                        session_id,
                        index,
                        message.role,
                        message.content,
                        message.status,
                        message.verdict.route if message.verdict else None,
                        message.verdict.model if message.verdict else None,
                        message.verdict.latencyMs if message.verdict else None,
                    )
                    for index, message in enumerate(payload.messages)
                ],
            )
        db.commit()
    except psycopg2.Error as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save chat session") from exc

    return _load_session(db, session_id)


def _load_session(db: PgConnection, session_id: UUID) -> ChatSessionDetail:
    with db.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            select
                id,
                title,
                preview,
                updated_at as "updatedAt"
            from chat_sessions
            where id = %s
            """,
            (session_id,),
        )
        session = cursor.fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")

        cursor.execute(
            """
            select id, role, content, status, route, model, latency_ms
            from chat_messages
            where session_id = %s
            order by position asc
            """,
            (session_id,),
        )
        messages = cursor.fetchall()

    return ChatSessionDetail(
        **session,
        messages=[_message_to_schema(message) for message in messages],
    )


def _message_to_schema(message: dict) -> ChatMessageSchema:
    verdict = None
    if message["route"] and message["model"]:
        verdict = {
            "route": message["route"],
            "model": message["model"],
            "latencyMs": message["latency_ms"],
        }
    return ChatMessageSchema(
        id=message["id"],
        role=message["role"],
        content=message["content"],
        status=message["status"],
        verdict=verdict,
    )
