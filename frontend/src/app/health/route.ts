export async function GET() {
  return Response.json({
    status: "ok",
    database: "not_configured",
    apiVersion: "v1",
    requestId: crypto.randomUUID(),
  });
}
