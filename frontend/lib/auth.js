/**
 * JWT authentication helpers (jose — works in Edge runtime).
 */
import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";

const SECRET = new TextEncoder().encode(
    process.env.JWT_SECRET || "xray3d_default_secret"
);
const ALG = "HS256";
const COOKIE_NAME = "boneview_token";
const EXPIRES_IN = "24h";

/** Create a signed JWT for the given user payload. */
export async function signToken(payload) {
    return new SignJWT(payload)
        .setProtectedHeader({ alg: ALG })
        .setIssuedAt()
        .setExpirationTime(EXPIRES_IN)
        .sign(SECRET);
}

/** Verify a JWT and return the payload, or null. */
export async function verifyToken(token) {
    try {
        const { payload } = await jwtVerify(token, SECRET);
        return payload;
    } catch {
        return null;
    }
}

/** Read the token from the request cookie and verify it. */
export async function getSession() {
    const cookieStore = await cookies();
    const token = cookieStore.get(COOKIE_NAME)?.value;
    if (!token) return null;
    return verifyToken(token);
}

export { COOKIE_NAME };
