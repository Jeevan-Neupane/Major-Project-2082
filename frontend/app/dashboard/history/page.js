"use client";

import { useState, useEffect } from "react";

export default function HistoryPage() {
    const [scans, setScans] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/results")
            .then((r) => r.json())
            .then((d) => setScans(Array.isArray(d) ? d : []))
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="max-w-7xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-100 mb-1">Scan History</h1>
            <p className="text-gray-500 text-sm mb-6">
                All 3D bone reconstruction records
            </p>

            {loading ? (
                <div className="text-center py-16 text-gray-500">Loading...</div>
            ) : scans.length === 0 ? (
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-12 text-center">
                    <svg className="w-16 h-16 mx-auto mb-4 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                            d="M9 12h6m-3-3v6m-7 4h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                    <p className="text-gray-500">No scans yet</p>
                    <a href="/dashboard/predict"
                        className="inline-block mt-3 text-primary-400 hover:text-primary-300 text-sm font-medium">
                        Order first scan &rarr;
                    </a>
                </div>
            ) : (
                <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-gray-500 text-xs uppercase tracking-wider">
                                    <th className="px-4 py-3">Date</th>
                                    <th className="px-4 py-3">Patient ID</th>
                                    <th className="px-4 py-3">Patient</th>
                                    <th className="px-4 py-3">Age / Gender</th>
                                    <th className="px-4 py-3">Anatomy</th>
                                    <th className="px-4 py-3">Model</th>
                                    <th className="px-4 py-3">Files</th>
                                    <th className="px-4 py-3">Mesh</th>
                                    <th className="px-4 py-3">Doctor</th>
                                    <th className="px-4 py-3">Status</th>
                                    <th className="px-4 py-3">3D</th>
                                </tr>
                            </thead>
                            <tbody>
                                {scans.map((s) => (
                                    <tr key={s._id} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                                        <td className="px-4 py-3 text-gray-400 whitespace-nowrap text-xs">
                                            {new Date(s.createdAt).toLocaleDateString("en-US", {
                                                month: "short", day: "numeric", year: "numeric",
                                            })}
                                            <br />
                                            <span className="text-gray-600">
                                                {new Date(s.createdAt).toLocaleTimeString("en-US", {
                                                    hour: "2-digit", minute: "2-digit",
                                                })}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-xs font-mono text-gray-500">
                                            {s.patient?.patientId || "—"}
                                        </td>
                                        <td className="px-4 py-3 font-medium text-gray-200">
                                            {s.patient?.name || "—"}
                                        </td>
                                        <td className="px-4 py-3 text-gray-500 text-xs">
                                            {s.patient?.age ? `${s.patient.age}y` : "—"} / {s.patient?.gender || "—"}
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="inline-block px-2 py-0.5 rounded-full bg-blue-900/30 text-blue-400 text-xs font-medium capitalize">
                                                {s.anatomy || "—"}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="inline-block px-2 py-0.5 rounded-full bg-primary-900/40 text-primary-400 text-xs font-medium">
                                                {s.modelKey}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-xs text-gray-500">
                                            <div>{s.apFileName || "—"}</div>
                                            {s.latFileName && <div className="text-gray-600">{s.latFileName}</div>}
                                        </td>
                                        <td className="px-4 py-3 text-xs text-gray-500">
                                            {s.vertexCount ? `${s.vertexCount.toLocaleString()} v` : "—"}
                                            {s.faceCount ? ` / ${s.faceCount.toLocaleString()} f` : ""}
                                        </td>
                                        <td className="px-4 py-3 text-gray-500 text-xs">
                                            {s.doctor?.name || "—"}
                                        </td>
                                        <td className="px-4 py-3">
                                            <StatusBadge status={s.status} />
                                        </td>
                                        <td className="px-4 py-3">
                                            {s.meshFilePath ? (
                                                <a
                                                    href={`/dashboard/history/${s._id}`}
                                                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-primary-900/40 text-primary-400 text-xs font-medium hover:bg-primary-900/60 transition-colors"
                                                >
                                                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
                                                    </svg>
                                                    View
                                                </a>
                                            ) : (
                                                <span className="text-gray-600 text-xs">—</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}

function StatusBadge({ status }) {
    const styles = {
        completed: "bg-green-900/30 text-green-400",
        processing: "bg-yellow-900/30 text-yellow-400",
        pending: "bg-gray-800 text-gray-500",
        failed: "bg-red-900/30 text-red-400",
    };
    return (
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize ${styles[status] || styles.pending}`}>
            {status || "—"}
        </span>
    );
}
