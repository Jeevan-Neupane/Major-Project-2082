"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
    const router = useRouter();
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    async function handleSubmit(e) {
        e.preventDefault();
        setError("");
        setLoading(true);

        try {
            const res = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password }),
            });
            const data = await res.json();

            if (!res.ok) {
                setError(data.error || "Login failed");
                return;
            }

            // Redirect based on role
            if (data.user.role === "admin") {
                router.push("/admin");
            } else {
                router.push("/dashboard");
            }
        } catch {
            setError("Network error. Please try again.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen flex">
            {/* Left panel — branding */}
            <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-primary-700 via-primary-600 to-primary-800 items-center justify-center p-12">
                <div className="max-w-md text-white">
                    <div className="w-16 h-16 rounded-2xl bg-white/20 flex items-center justify-center mb-8">
                        <svg className="w-9 h-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
                        </svg>
                    </div>
                    <h1 className="text-4xl font-bold mb-4">BoneView</h1>
                    <p className="text-xl text-white/90 mb-6">
                        X-Ray to 3D Bone Reconstruction
                    </p>
                    <p className="text-sm text-white/70 leading-relaxed">
                        Upload standard X-ray images and get instant 3D bone reconstructions
                        with clinical measurements. Powered by deep learning models trained
                        on thousands of CT scans.
                    </p>
                    <div className="mt-10 grid grid-cols-3 gap-4 text-center">
                        <div className="bg-white/10 rounded-lg p-3">
                            <p className="text-2xl font-bold">12</p>
                            <p className="text-xs text-white/60">AI Models</p>
                        </div>
                        <div className="bg-white/10 rounded-lg p-3">
                            <p className="text-2xl font-bold">6</p>
                            <p className="text-xs text-white/60">Anatomies</p>
                        </div>
                        <div className="bg-white/10 rounded-lg p-3">
                            <p className="text-2xl font-bold">3D</p>
                            <p className="text-xs text-white/60">Live View</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Right panel — form */}
            <div className="flex-1 flex items-center justify-center p-8 bg-gray-950">
                <div className="w-full max-w-sm">
                    <div className="lg:hidden mb-8 text-center">
                        <h1 className="text-3xl font-bold text-primary-400">BoneView</h1>
                        <p className="text-sm text-gray-500">X-Ray to 3D Reconstruction</p>
                    </div>

                    <h2 className="text-2xl font-bold text-gray-100 mb-1">Welcome back</h2>
                    <p className="text-sm text-gray-500 mb-8">Sign in to your account</p>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">
                                Username
                            </label>
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="w-full px-4 py-2.5 rounded-lg border border-gray-700 bg-gray-900 text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-shadow placeholder-gray-600"
                                placeholder="Enter username"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">
                                Password
                            </label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full px-4 py-2.5 rounded-lg border border-gray-700 bg-gray-900 text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-shadow placeholder-gray-600"
                                placeholder="Enter password"
                                required
                            />
                        </div>

                        {error && (
                            <div className="bg-red-900/30 text-red-400 text-sm px-4 py-2.5 rounded-lg border border-red-800">
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-2.5 px-4 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-950 transition-colors disabled:opacity-50"
                        >
                            {loading ? "Signing in..." : "Sign In"}
                        </button>
                    </form>

                    <p className="mt-8 text-xs text-gray-600 text-center">
                        Default admin: <strong>admin</strong> / <strong>admin123</strong>
                    </p>
                </div>
            </div>
        </div>
    );
}
