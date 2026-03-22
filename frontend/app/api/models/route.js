import { NextResponse } from "next/server";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

/** GET /api/models → proxy to FastAPI /models */
export async function GET() {
    try {
        const res = await fetch(`${API}/models`, { cache: "no-store" });
        const data = await res.json();
        return NextResponse.json(data);
    } catch (err) {
        return NextResponse.json(
            { error: "Backend unreachable", detail: err.message },
            { status: 502 }
        );
    }
}
