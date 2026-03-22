import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getScans } from "@/lib/db";

export async function GET() {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const scans =
        session.role === "admin"
            ? await getScans()
            : await getScans(session.id);

    return NextResponse.json(scans);
}
