import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getScanById } from "@/lib/db";
import fs from "fs/promises";
import path from "path";

const SCAN_DATA_DIR = path.join(process.cwd(), "scan_data");

/** GET /api/scans/[id]/mesh — return saved mesh JSON for 3D viewer */
export async function GET(request, { params }) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const scan = await getScanById(params.id);
    if (!scan) {
        return NextResponse.json({ error: "Scan not found" }, { status: 404 });
    }

    if (!scan.meshFilePath) {
        return NextResponse.json({ error: "No 3D data saved for this scan" }, { status: 404 });
    }

    try {
        const filePath = path.join(SCAN_DATA_DIR, scan.meshFilePath);
        const data = await fs.readFile(filePath, "utf-8");
        return new NextResponse(data, {
            headers: { "Content-Type": "application/json" },
        });
    } catch {
        return NextResponse.json({ error: "Mesh file not found on disk" }, { status: 404 });
    }
}
