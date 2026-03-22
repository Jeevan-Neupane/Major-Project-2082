import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getScanById, updateScanDiagnosis } from "@/lib/db";

/** GET /api/scans/[id] — single scan detail */
export async function GET(request, { params }) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const scan = await getScanById(params.id);
    if (!scan) {
        return NextResponse.json({ error: "Scan not found" }, { status: 404 });
    }

    return NextResponse.json(scan);
}

/** PATCH /api/scans/[id] — update diagnosis/notes */
export async function PATCH(request, { params }) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const updated = await updateScanDiagnosis(params.id, {
        diagnosis: body.diagnosis || "",
        notes: body.notes || "",
    });

    if (!updated) {
        return NextResponse.json({ error: "Scan not found" }, { status: 404 });
    }

    return NextResponse.json(updated);
}
