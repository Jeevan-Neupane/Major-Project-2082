import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getUsers, addUser, deleteUser } from "@/lib/db";

/** GET /api/admin/doctors → list all users (admin only) */
export async function GET() {
    const session = await getSession();
    if (!session || session.role !== "admin") {
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const users = (await getUsers()).map(({ passwordHash, ...u }) => u);
    return NextResponse.json(users);
}

/** POST /api/admin/doctors → create new doctor */
export async function POST(request) {
    const session = await getSession();
    if (!session || session.role !== "admin") {
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const body = await request.json();
    const { username, name, password, role, speciality, phone } = body;

    if (!username || !name || !password) {
        return NextResponse.json(
            { error: "username, name, and password are required" },
            { status: 400 }
        );
    }

    const result = await addUser({
        username,
        name,
        password,
        role: role || "doctor",
        speciality: speciality || "",
        phone: phone || "",
    });

    if (result.error) {
        return NextResponse.json({ error: result.error }, { status: 409 });
    }

    const { passwordHash, ...safe } = result.user;
    return NextResponse.json({ user: safe }, { status: 201 });
}

/** DELETE /api/admin/doctors → remove doctor by id in JSON body */
export async function DELETE(request) {
    const session = await getSession();
    if (!session || session.role !== "admin") {
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const body = await request.json();
    const id = body.id;
    if (!id) {
        return NextResponse.json({ error: "id is required" }, { status: 400 });
    }

    const result = await deleteUser(id);
    if (result.error) {
        return NextResponse.json({ error: result.error }, { status: 400 });
    }

    return NextResponse.json({ ok: true });
}
