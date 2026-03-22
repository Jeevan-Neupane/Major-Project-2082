import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getDashboardStats } from "@/lib/db";

/** GET /api/stats — dashboard statistics */
export async function GET() {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const stats =
        session.role === "admin"
            ? await getDashboardStats()
            : await getDashboardStats(session.id);

    return NextResponse.json(stats);
}
