"use client";

import { useState, useEffect } from "react";

export default function PatientsPage() {
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");

    useEffect(() => {
        const url = search.length >= 2
            ? `/api/patients?q=${encodeURIComponent(search)}`
            : "/api/patients";

        const timeout = setTimeout(() => {
            fetch(url)
                .then((r) => r.json())
                .then((d) => setPatients(Array.isArray(d) ? d : []))
                .catch(() => { })
                .finally(() => setLoading(false));
        }, search ? 300 : 0);

        return () => clearTimeout(timeout);
    }, [search]);

    return (
        <div className="max-w-6xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-100 mb-1">Patient Registry</h1>
            <p className="text-gray-500 text-sm mb-6">All registered patients</p>

            {/* Search */}
            <div className="mb-4">
                <input
                    type="text"
                    placeholder="Search by name or Patient ID..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full max-w-md px-3 py-2 rounded-lg border border-gray-700 bg-gray-900 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
                />
            </div>

            {loading ? (
                <div className="text-center py-16 text-gray-500">Loading...</div>
            ) : patients.length === 0 ? (
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-12 text-center">
                    <p className="text-gray-500">No patients found</p>
                    <a href="/dashboard/predict" className="inline-block mt-3 text-primary-400 hover:text-primary-300 text-sm font-medium">
                        Register a patient via New Scan &rarr;
                    </a>
                </div>
            ) : (
                <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-gray-500 text-xs uppercase tracking-wider">
                                <th className="px-5 py-3">Patient ID</th>
                                <th className="px-5 py-3">Name</th>
                                <th className="px-5 py-3">Age</th>
                                <th className="px-5 py-3">Gender</th>
                                <th className="px-5 py-3">Phone</th>
                                <th className="px-5 py-3">Registered</th>
                                <th className="px-5 py-3">Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {patients.map((p) => (
                                <tr key={p._id} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                                    <td className="px-5 py-3 font-mono text-xs text-gray-500">{p.patientId}</td>
                                    <td className="px-5 py-3 font-medium text-gray-200">{p.name}</td>
                                    <td className="px-5 py-3 text-gray-400">{p.age || "—"}</td>
                                    <td className="px-5 py-3 text-gray-400">{p.gender || "—"}</td>
                                    <td className="px-5 py-3 text-gray-400">{p.phone || "—"}</td>
                                    <td className="px-5 py-3 text-gray-500 text-xs">
                                        {new Date(p.createdAt).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                                    </td>
                                    <td className="px-5 py-3 text-gray-500 text-xs max-w-[200px] truncate">{p.notes || "—"}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
