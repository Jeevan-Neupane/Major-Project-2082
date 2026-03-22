import { NextResponse } from "next/server";
import { getSession, COOKIE_NAME } from "@/lib/auth";

export async function GET() {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ user: null }, { status: 401 });
    }
    return NextResponse.json({
        user: {
            id: session.id,
            username: session.username,
            name: session.name,
            role: session.role,
        },
    });
}

/** POST /api/auth/me  → logout */
export async function POST() {
    const res = NextResponse.json({ ok: true });
    res.cookies.delete(COOKIE_NAME);
    return res;
}
