const FASTAPI_BASE = (
  process.env.FASTAPI_BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://165.245.131.71:8000"
).replace(/\/$/, "");

export async function GET() {
  try {
    const upstream = await fetch(`${FASTAPI_BASE}/dashboard/usage`, {
      method: "GET",
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
