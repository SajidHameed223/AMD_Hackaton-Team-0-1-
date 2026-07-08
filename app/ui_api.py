from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter
import psycopg2
from psycopg2.extras import RealDictCursor

from app.database import get_connection
from app.schemas import ChatRequest, ChatResponse, UsageSummary


router = APIRouter(tags=["ui"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    reply = _pick_reply(payload.message)
    return ChatResponse(**reply)


@router.get("/usage", response_model=UsageSummary)
def usage() -> UsageSummary:
    return _build_usage_summary()


def _pick_reply(message: str) -> dict:
    text = message.lower()
    if any(term in text for term in ("big-o", "big o", "complexity", "notation")):
        return {
            "route": "cloud",
            "model": "qwen-72b",
            "latency_ms": 840,
            "reply": (
                "Big-O describes how work grows as input grows. A hash lookup is "
                "**O(1)** because the expected lookup work stays flat, while a "
                "nested pairwise comparison is usually **O(n^2)** because the work "
                "grows with every pair."
            ),
        }
    if any(term in text for term in ("code", "python", "function", "script", "sort")):
        return {
            "route": "cloud",
            "model": "qwen-72b",
            "latency_ms": 1040,
            "reply": (
                "The router marked this as a coding task, so auto-routing sent it "
                "to the larger model. The frontend can replace this stubbed reply "
                "with the real model response without changing the contract."
            ),
        }
    route = "cloud" if len(message) > 90 else "local"
    return {
        "route": route,
        "model": "qwen-72b" if route == "cloud" else "llama-3.1-8b",
        "latency_ms": 900 if route == "cloud" else 95,
        "reply": (
            "Auto-routing is connected through FastAPI. This endpoint returns the "
            "same shape the frontend expects, so backend model wiring can slot in "
            "behind it cleanly."
        ),
    }


def _build_usage_summary() -> UsageSummary:
    today = datetime.now(UTC).date()
    days = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
    labels = [_format_day(day) for day in days]
    empty = _empty_usage(labels)

    connection = None
    try:
        connection = get_connection()
        if connection is None:
            return empty
        rows = _fetch_usage_rows(connection, days[0])
    except psycopg2.Error:
        return empty
    finally:
        if connection is not None:
            connection.close()

    if not rows:
        return empty

    day_index = {day: index for index, day in enumerate(days)}
    input_tokens = [0] * len(days)
    output_tokens = [0] * len(days)
    route_points = {
        "llama-3.1-8b": [0] * len(days),
        "qwen-72b": [0] * len(days),
    }
    cost_trend = [0.0] * len(days)
    model_rows: dict[str, dict] = {}
    request_trend = [0] * len(days)

    for row in rows:
        created_at = row["created_at"]
        row_day = created_at.date() if isinstance(created_at, datetime) else created_at
        index = day_index.get(row_day)
        if index is None:
            continue

        tokens = _estimate_tokens(row["content"])
        if row["role"] == "user":
            input_tokens[index] += tokens
            continue

        route = row["route"] or "local"
        model = row["model"] or ("qwen-72b" if route == "cloud" else "llama-3.1-8b")
        output_tokens[index] += tokens
        request_trend[index] += 1
        route_points.setdefault(model, [0] * len(days))[index] += tokens
        model_row = model_rows.setdefault(
            model,
            {
                "model": model,
                "provider": route,
                "requests": 0,
                "inputTokens": 0,
                "outputTokens": 0,
                "cost": 0.0,
            },
        )
        model_row["requests"] += 1
        model_row["outputTokens"] += tokens
        if route == "cloud":
            cost = tokens * 0.000015
            model_row["cost"] += cost
            cost_trend[index] += cost

    assistant_requests = sum(request_trend)
    total_tokens = sum(input_tokens) + sum(output_tokens)
    local_requests = sum(
        row["requests"] for row in model_rows.values() if row["provider"] == "local"
    )
    cloud_cost = sum(row["cost"] for row in model_rows.values())
    local_rate = round((local_requests / assistant_requests) * 100) if assistant_requests else 0

    return UsageSummary(
        rangeLabel=f"{labels[0]} - {labels[-1]} · Postgres history",
        days=labels,
        routeSeries=[
            {"name": model, "points": points}
            for model, points in route_points.items()
            if any(points) or model in {"llama-3.1-8b", "qwen-72b"}
        ],
        tokenInput=input_tokens,
        tokenOutput=output_tokens,
        modelRows=sorted(model_rows.values(), key=lambda row: row["model"]),
        stats={
            "requests": {
                "value": f"{assistant_requests:,}",
                "delta": {"text": "live from Postgres", "direction": "flat", "good": True},
                "trend": request_trend,
            },
            "tokens": {
                "value": _compact(total_tokens),
                "delta": {"text": "estimated from text", "direction": "flat", "good": True},
                "trend": [input_tokens[i] + output_tokens[i] for i in range(len(days))],
            },
            "localRate": {
                "value": f"{local_rate}%",
                "delta": {"text": "auto-routing only", "direction": "flat", "good": True},
                "trend": [local_rate] * len(days),
            },
            "cost": {
                "value": f"${cloud_cost:.2f}",
                "delta": {"text": "estimated cloud spend", "direction": "flat", "good": True},
                "trend": [round(cost, 4) for cost in cost_trend],
            },
        },
        budget={"spent": round(cloud_cost, 2), "limit": 25.0, "paceLabel": f"${cloud_cost:.2f}"},
    )


def _fetch_usage_rows(connection, start_day: date) -> list[dict]:
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            select created_at, role, content, route, model
            from chat_messages
            where created_at >= %s
            order by created_at asc
            """,
            (datetime.combine(start_day, datetime.min.time(), tzinfo=UTC),),
        )
        return list(cursor.fetchall())


def _empty_usage(labels: list[str]) -> UsageSummary:
    zeroes = [0] * len(labels)
    return UsageSummary(
        rangeLabel=f"{labels[0]} - {labels[-1]} · no Postgres history yet",
        days=labels,
        routeSeries=[
            {"name": "llama-3.1-8b", "points": zeroes},
            {"name": "qwen-72b", "points": zeroes},
        ],
        tokenInput=zeroes,
        tokenOutput=zeroes,
        modelRows=[],
        stats={
            "requests": {
                "value": "0",
                "delta": {"text": "no saved chats yet", "direction": "flat", "good": True},
                "trend": zeroes,
            },
            "tokens": {
                "value": "0",
                "delta": {"text": "no saved chats yet", "direction": "flat", "good": True},
                "trend": zeroes,
            },
            "localRate": {
                "value": "0%",
                "delta": {"text": "auto-routing only", "direction": "flat", "good": True},
                "trend": zeroes,
            },
            "cost": {
                "value": "$0.00",
                "delta": {"text": "no cloud spend yet", "direction": "flat", "good": True},
                "trend": zeroes,
            },
        },
        budget={"spent": 0.0, "limit": 25.0, "paceLabel": "$0.00"},
    )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _format_day(day: date) -> str:
    return f"{day.strftime('%b')} {day.day}"


def _compact(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"
