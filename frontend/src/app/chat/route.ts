const FASTAPI_BASE = (
  process.env.FASTAPI_BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const upstream = await fetch(`${FASTAPI_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
    });

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return Response.json({ detail: "FastAPI backend unavailable" }, { status: 502 });
  }
}
