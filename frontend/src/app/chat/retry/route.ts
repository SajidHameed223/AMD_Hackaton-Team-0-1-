import { answerChat } from "@/lib/server/track1Router";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    return Response.json(await answerChat(payload));
  } catch {
    return Response.json({ detail: "Invalid retry request" }, { status: 400 });
  }
}
