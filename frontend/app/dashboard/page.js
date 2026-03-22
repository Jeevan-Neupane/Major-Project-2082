"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function DashboardPage() {
    const [user, setUser] = useState(null);
    const [models, setModels] = useState([]);
    const [stats, setStats] = useState(null);
    const [backendOk, setBackendOk] = useState(null);

    useEffect(() => {
        fetch("/api/auth/me")
            .then((r) => r.json())
            .then((d) => setUser(d.user))
            .catch(() => { });

        fetch("/api/models")
            .then((r) => r.json())
            .then((d) => {
                if (Array.isArray(d)) {
                    setModels(d);
                    setBackendOk(true);
                } else {
                    setBackendOk(false);
                }
            })
            .catch(() => setBackendOk(false));

        fetch("/api/stats")
            .then((r) => r.json())
            .then((d) => setStats(d))
            .catch(() => { });
    }, []);

    const available = models.filter((m) => m.weights_exist);

    return (
        <div className="max-w-6xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-100 mb-1">
                Welcome{user ? `, ${user.name}` : ""}
            </h1>
            <p className="text-gray-500 text-sm mb-8">
                Hospital radiology dashboard &mdash; 3D bone reconstruction
            </p>

            {/* Status Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
                <Card
                    label="AI Backend"
                    value={backendOk === null ? "..." : backendOk ? "Online" : "Offline"}
                    color={backendOk ? "green" : backendOk === false ? "red" : "gray"}
                    icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008z" /></svg>}
                />
                <Card label="AI Models" value={available.length} color="blue"
                    icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>}
                />
                <Card label="Total Scans" value={stats?.totalScans ?? "..."} color="purple"
                    icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" /></svg>}
                />
                <Card label="Patients" value={stats?.totalPatients ?? "..."} color="amber"
                    icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>}
                />
                <Card label="Role" value={user?.role === "admin" ? "Admin" : "Doctor"} color="gray"
                    icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" /></svg>}
                />
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
                <Link href="/dashboard/predict"
                    className="group bg-gray-900 rounded-xl border border-gray-800 p-5 hover:border-primary-700 hover:shadow-md hover:shadow-primary-900/20 transition-all">
                    <div className="w-10 h-10 rounded-lg bg-primary-900/40 flex items-center justify-center mb-3 group-hover:bg-primary-900/60">
                        <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    </div>
                    <h3 className="font-semibold text-gray-100 text-sm">New Scan Order</h3>
                    <p className="text-xs text-gray-500 mt-1">Register patient & upload X-rays</p>
                </Link>

                <Link href="/dashboard/history"
                    className="group bg-gray-900 rounded-xl border border-gray-800 p-5 hover:border-gray-700 hover:shadow-md transition-all">
                    <div className="w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center mb-3 group-hover:bg-gray-700">
                        <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    </div>
                    <h3 className="font-semibold text-gray-100 text-sm">Scan History</h3>
                    <p className="text-xs text-gray-500 mt-1">Browse all reconstruction records</p>
                </Link>

                <Link href="/dashboard/patients"
                    className="group bg-gray-900 rounded-xl border border-gray-800 p-5 hover:border-gray-700 hover:shadow-md transition-all">
                    <div className="w-10 h-10 rounded-lg bg-amber-900/30 flex items-center justify-center mb-3 group-hover:bg-amber-900/50">
                        <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" /></svg>
                    </div>
                    <h3 className="font-semibold text-gray-100 text-sm">Patient Registry</h3>
                    <p className="text-xs text-gray-500 mt-1">View all registered patients</p>
                </Link>
            </div>

            {/* Recent Scans */}
            {stats?.recentScans?.length > 0 && (
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 mb-8">
                    <h2 className="font-semibold text-gray-100 mb-3">Recent Scans</h2>
                    <div className="space-y-2">
                        {stats.recentScans.map((s) => (
                            <div key={s._id} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0 text-sm">
                                <div className="flex items-center gap-3">
                                    <span className="w-2 h-2 rounded-full bg-green-400" />
                                    <span className="font-medium text-gray-200">{s.patient?.name || "Unknown"}</span>
                                    <span className="text-xs text-gray-500">{s.patient?.patientId}</span>
                                </div>
                                <div className="flex items-center gap-4 text-xs text-gray-500">
                                    <span className="px-2 py-0.5 rounded-full bg-primary-900/40 text-primary-400 font-medium">{s.modelKey}</span>
                                    <span>{s.doctor?.name}</span>
                                    <span>{new Date(s.createdAt).toLocaleDateString()}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Available Models */}
            {available.length > 0 && (
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
                    <h2 className="font-semibold text-gray-100 mb-3">Available AI Models</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        {available.map((m) => (
                            <div key={m.model_key} className="flex items-center gap-3 p-3 rounded-lg bg-gray-800 border border-gray-700">
                                <div className={`w-2 h-2 rounded-full ${m.model_key.startsWith("swinx2s") ? "bg-primary-500" : "bg-emerald-500"}`} />
                                <div>
                                    <p className="text-sm font-medium text-gray-200">{m.model_key}</p>
                                    <p className="text-xs text-gray-500">{m.description}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

function Card({ label, value, color, icon }) {
    const colors = {
        green: "bg-green-900/30 text-green-400 border-green-800",
        red: "bg-red-900/30 text-red-400 border-red-800",
        blue: "bg-blue-900/30 text-blue-400 border-blue-800",
        purple: "bg-purple-900/30 text-purple-400 border-purple-800",
        amber: "bg-amber-900/30 text-amber-400 border-amber-800",
        gray: "bg-gray-900 text-gray-400 border-gray-800",
    };
    return (
        <div className={`rounded-xl border p-4 ${colors[color] || colors.gray}`}>
            <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-medium opacity-80">{label}</p>
                <span className="opacity-60">{icon}</span>
            </div>
            <p className="text-2xl font-bold">{value}</p>
        </div>
    );
}
