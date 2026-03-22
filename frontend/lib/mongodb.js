import mongoose from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/boneview";

if (!MONGODB_URI) {
    throw new Error("Please define MONGODB_URI in .env.local");
}

/**
 * Global cache so hot-reloads in dev don't create multiple connections.
 */
let cached = global._mongooseCache;
if (!cached) {
    cached = global._mongooseCache = { conn: null, promise: null };
}

export default async function dbConnect() {
    if (cached.conn) return cached.conn;

    if (!cached.promise) {
        cached.promise = mongoose
            .connect(MONGODB_URI, { bufferCommands: false })
            .then((m) => m);
    }

    cached.conn = await cached.promise;
    return cached.conn;
}
