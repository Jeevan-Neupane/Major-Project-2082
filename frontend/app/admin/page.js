"use client";

import { useState, useEffect, useCallback } from "react";

export default function AdminPage() {
    const [doctors, setDoctors] = useState([]);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

    // Form state
    const [name, setName] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [speciality, setSpeciality] = useState("");

    const fetchDoctors = useCallback(() => {
        fetch("/api/admin/doctors")
            .then((r) => r.json())
            .then((d) => setDoctors(Array.isArray(d) ? d : []))
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    useEffect(() => {
        fetchDoctors();
    }, [fetchDoctors]);

    async function handleAdd(e) {
        e.preventDefault();
        setError("");
        setSuccess("");
        setSubmitting(true);

        try {
            const res = await fetch("/api/admin/doctors", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, username, password, speciality }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || "Failed to add doctor");
            }

            setSuccess(`Doctor "${name}" added successfully`);
            setName("");
            setUsername("");
            setPassword("");
            setSpeciality("");
            fetchDoctors();
        } catch (err) {
            setError(err.message);
        } finally {
            setSubmitting(false);
        }
    }

    async function handleDelete(id, doctorName) {
        if (!window.confirm(`Remove doctor "${doctorName}"?`)) return;

        try {
            const res = await fetch("/api/admin/doctors", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id }),
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || "Failed to remove doctor");
            }

            setSuccess(`Doctor "${doctorName}" removed`);
            fetchDoctors();
        } catch (err) {
            setError(err.message);
        }
    }

    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-100 mb-1">
                Manage Doctors
            </h1>
            <p className="text-gray-500 text-sm mb-6">
                Add or remove doctors who can access the system
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* Add Doctor Form */}
                <form
                    onSubmit={handleAdd}
                    className="lg:col-span-2 bg-gray-900 rounded-xl border border-gray-800 p-6 space-y-4 h-fit"
                >
                    <h2 className="font-semibold text-gray-100">Add New Doctor</h2>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Full Name
                        </label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Dr. Jane Smith"
                            required
                            className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Username
                        </label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="jsmith"
                            required
                            className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
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
                            placeholder="••••••••"
                            required
                            minLength={6}
                            className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Speciality
                        </label>
                        <input
                            type="text"
                            value={speciality}
                            onChange={(e) => setSpeciality(e.target.value)}
                            placeholder="e.g. Orthopaedics, Radiology"
                            className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
                        />
                    </div>

                    {error && (
                        <div className="bg-red-900/30 text-red-400 text-sm px-4 py-2 rounded-lg border border-red-800">
                            {error}
                        </div>
                    )}
                    {success && (
                        <div className="bg-green-900/30 text-green-400 text-sm px-4 py-2 rounded-lg border border-green-800">
                            {success}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={submitting}
                        className="w-full py-2.5 bg-primary-600 text-white rounded-lg font-medium text-sm hover:bg-primary-700 transition-colors disabled:opacity-50"
                    >
                        {submitting ? "Adding..." : "Add Doctor"}
                    </button>
                </form>

                {/* Doctors List */}
                <div className="lg:col-span-3 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                    <div className="px-5 py-4 border-b border-gray-800">
                        <h2 className="font-semibold text-gray-100">
                            Doctors{" "}
                            <span className="text-gray-500 font-normal text-sm">
                                ({doctors.filter((d) => d.role === "doctor").length})
                            </span>
                        </h2>
                    </div>

                    {loading ? (
                        <div className="p-8 text-center text-gray-500 text-sm">
                            Loading...
                        </div>
                    ) : doctors.filter((d) => d.role === "doctor").length === 0 ? (
                        <div className="p-8 text-center text-gray-500 text-sm">
                            No doctors added yet
                        </div>
                    ) : (
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-left text-xs text-gray-500 uppercase tracking-wider border-b border-gray-800 bg-gray-900/80">
                                    <th className="px-5 py-3">Name</th>
                                    <th className="px-5 py-3">Username</th>
                                    <th className="px-5 py-3">Speciality</th>
                                    <th className="px-5 py-3">Added</th>
                                    <th className="px-5 py-3"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {doctors
                                    .filter((d) => d.role === "doctor")
                                    .map((doc) => (
                                        <tr
                                            key={doc._id}
                                            className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors"
                                        >
                                            <td className="px-5 py-3 font-medium text-gray-200">
                                                {doc.name}
                                            </td>
                                            <td className="px-5 py-3 text-gray-400">
                                                {doc.username}
                                            </td>
                                            <td className="px-5 py-3 text-gray-500 text-xs">
                                                {doc.speciality || "—"}
                                            </td>
                                            <td className="px-5 py-3 text-gray-500 text-xs">
                                                {doc.createdAt
                                                    ? new Date(doc.createdAt).toLocaleDateString()
                                                    : "—"}
                                            </td>
                                            <td className="px-5 py-3 text-right">
                                                <button
                                                    onClick={() => handleDelete(doc._id, doc.name)}
                                                    className="text-red-400 hover:text-red-300 text-xs font-medium"
                                                >
                                                    Remove
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}
