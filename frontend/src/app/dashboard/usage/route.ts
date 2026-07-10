import { demoUsage } from "@/lib/server/track1Router";

export async function GET() {
  return Response.json(demoUsage());
}
