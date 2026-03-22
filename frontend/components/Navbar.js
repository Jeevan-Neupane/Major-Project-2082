"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function Navbar() {
    const router = useRouter();
    const [user, setUser] = useState(null);

    useEffect(() => {
        fetch("/api/auth/me")
            .then((r) => (r.ok ? r.json() : null))
            .then((d) => d && setUser(d.user))
            .catch(() => { });
    }, []);

    async function handleLogout() {
        await fetch("/api/auth/me", { method: "POST" });
        router.push("/login");
    }

    return (
        <header className="h-16 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6 shadow-lg">
            <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-primary-600 flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
                    </svg>
                </div>
                <span className="text-xl font-bold text-gray-100">BoneView</span>
                <span className="text-xs text-gray-500 ml-1 hidden sm:inline">X-Ray → 3D</span>
            </div>

            {user && (
                <div className="flex items-center gap-4">
                    <div className="text-right hidden sm:block">
                        <p className="text-sm font-medium text-gray-200">{user.name}</p>
                        <p className="text-xs text-gray-500 capitalize">{user.role}</p>
                    </div>
                    <div className="w-9 h-9 rounded-full bg-primary-900/50 text-primary-400 flex items-center justify-center text-sm font-bold">
                        {user.name?.charAt(0).toUpperCase()}
                    </div>
                    <button
                        onClick={handleLogout}
                        className="text-sm text-gray-400 hover:text-red-400 transition-colors"
                    >
                        Logout
                    </button>
                </div>
            )}
        </header>
    );
}
