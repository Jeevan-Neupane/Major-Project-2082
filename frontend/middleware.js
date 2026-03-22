import { NextResponse } from "next/server";
import { jwtVerify } from "jose";

const SECRET = new TextEncoder().encode(
    process.env.JWT_SECRET || "xray3d_default_secret"
);

const PUBLIC_PATHS = ["/login", "/api/auth/login", "/_next", "/favicon.ico"];

export async function middleware(request) {
    const { pathname } = request.nextUrl;

    // Allow public paths
    if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
        return NextResponse.next();
    }

    const token = request.cookies.get("boneview_token")?.value;
    if (!token) {
        return NextResponse.redirect(new URL("/login", request.url));
    }

    try {
        const { payload } = await jwtVerify(token, SECRET);

        // Admin-only routes
        if (pathname.startsWith("/admin") && payload.role !== "admin") {
            return NextResponse.redirect(new URL("/dashboard", request.url));
        }

        // Attach user info via header for server components
        const res = NextResponse.next();
        res.headers.set("x-user-id", payload.id);
        res.headers.set("x-user-role", payload.role);
        return res;
    } catch {
        // Invalid token → back to login
        const res = NextResponse.redirect(new URL("/login", request.url));
        res.cookies.delete("boneview_token");
        return res;
    }
}

export const config = {
    matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
