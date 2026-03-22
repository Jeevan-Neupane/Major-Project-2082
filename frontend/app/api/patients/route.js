import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getPatients, searchPatients, createPatient } from "@/lib/db";

/** GET /api/patients?q=searchterm  — list or search patients */
export async function GET(request) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const q = searchParams.get("q");

    const patients = q
        ? await searchPatients(q)
        : session.role === "admin"
            ? await getPatients()
            : await getPatients(session.id);

    return NextResponse.json(patients);
}

/** POST /api/patients — register a new patient */
export async function POST(request) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { name, age, gender, phone, notes } = body;

    if (!name) {
        return NextResponse.json({ error: "Patient name is required" }, { status: 400 });
    }

    const patient = await createPatient({
        name,
        age: age ? parseInt(age, 10) : undefined,
        gender: gender || "",
        phone: phone || "",
        referredBy: session.id,
        notes: notes || "",
    });

    return NextResponse.json(patient, { status: 201 });
}
