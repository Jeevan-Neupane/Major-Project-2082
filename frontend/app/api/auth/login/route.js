import { NextResponse } from "next/server";
import { findUser, verifyPassword } from "@/lib/db";
import { signToken, COOKIE_NAME } from "@/lib/auth";

export async function POST(request) {
    const body = await request.json();
    const { username, password } = body;

    if (!username || !password) {
        return NextResponse.json(
            { error: "Username and password are required" },
            { status: 400 }
        );
    }

    const user = await findUser(username);
    if (!user || !verifyPassword(password, user.passwordHash)) {
        return NextResponse.json(
            { error: "Invalid username or password" },
            { status: 401 }
        );
    }

    const token = await signToken({
        id: user._id.toString(),
        username: user.username,
        name: user.name,
        role: user.role,
    });

    const res = NextResponse.json({
        user: { id: user._id.toString(), username: user.username, name: user.name, role: user.role },
    });

    res.cookies.set(COOKIE_NAME, token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        maxAge: 86400,
        path: "/",
    });

    return res;
}
