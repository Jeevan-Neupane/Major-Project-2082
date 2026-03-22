"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";

const BoneViewer = dynamic(() => import("@/components/BoneViewer"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-[600px] bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center">
            <p className="text-gray-500 text-sm">Loading 3D viewer...</p>
        </div>
    ),
});

export default function ScanDetailPage() {
    const { id } = useParams();
    const router = useRouter();
    const [scan, setScan] = useState(null);
    const [meshData, setMeshData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [meshLoading, setMeshLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        fetch(`/api/scans/${id}`)
            .then((r) => {
                if (!r.ok) throw new Error("Scan not found");
                return r.json();
            })
            .then((d) => {
                setScan(d);
                if (d.meshFilePath) {
                    setMeshLoading(true);
                    return fetch(`/api/scans/${id}/mesh`).then((r) => {
                        if (!r.ok) throw new Error("Mesh data not found");
                        return r.json();
                    });
                }
                return null;
            })
            .then((mesh) => {
                if (mesh) setMeshData(mesh);
            })
            .catch((err) => setError(err.message))
            .finally(() => {
                setLoading(false);
                setMeshLoading(false);
            });
    }, [id]);

    function handleDownloadOBJ() {
        if (!meshData) return;
        const lines = [];
        for (const v of meshData.vertices) {
            lines.push(`v ${v[0]} ${v[1]} ${v[2]}`);
        }
        if (meshData.normals) {
            for (const n of meshData.normals) {
                lines.push(`vn ${n[0]} ${n[1]} ${n[2]}`);
            }
        }
        for (const f of meshData.faces) {
            lines.push(`f ${f[0] + 1} ${f[1] + 1} ${f[2] + 1}`);
        }
        const blob = new Blob([lines.join("\n")], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `scan_${id}.obj`;
        a.click();
        URL.revokeObjectURL(url);
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-gray-500">
                Loading scan...
            </div>
        );
    }

    if (error) {
        return (
            <div className="max-w-4xl mx-auto">
                <div className="bg-red-900/30 text-red-400 text-sm px-4 py-3 rounded-lg border border-red-800">
                    {error}
                </div>
                <button onClick={() => router.back()} className="mt-4 text-sm text-primary-400 hover:text-primary-300">
                    &larr; Back to history
                </button>
            </div>
        );
    }

    return (
        <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <button onClick={() => router.push("/dashboard/history")} className="text-sm text-primary-400 hover:text-primary-300 mb-2 inline-block">
                        &larr; Back to Scan History
                    </button>
                    <h1 className="text-2xl font-bold text-gray-100">Scan Detail</h1>
                    <p className="text-gray-500 text-sm">{scan?.patient?.name} &mdash; {scan?.modelKey}</p>
                </div>
                <div className="flex items-center gap-2">
                    <a href="/dashboard/predict"
                        className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                        </svg>
                        New Scan
                    </a>
                    {meshData && (
                        <button
                            onClick={handleDownloadOBJ}
                            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            Download OBJ
                        </button>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 3D Viewer */}
                <div className="lg:col-span-2">
                    {meshLoading ? (
                        <div className="w-full h-[600px] bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center">
                            <p className="text-gray-500 text-sm">Loading 3D model...</p>
                        </div>
                    ) : meshData ? (
                        <BoneViewer meshData={meshData} />
                    ) : (
                        <div className="w-full h-[600px] bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center text-gray-600">
                            <p className="text-sm">No 3D data saved for this scan</p>
                        </div>
                    )}
                </div>

                {/* Scan Info */}
                <div className="space-y-4">
                    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-4">
                        <h3 className="font-semibold text-gray-100">Patient</h3>
                        <div className="space-y-2 text-sm">
                            <Info label="Name" value={scan?.patient?.name} />
                            <Info label="Patient ID" value={scan?.patient?.patientId} />
                            <Info label="Age" value={scan?.patient?.age ? `${scan.patient.age}y` : null} />
                            <Info label="Gender" value={scan?.patient?.gender} />
                        </div>
                    </div>

                    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-4">
                        <h3 className="font-semibold text-gray-100">Reconstruction</h3>
                        <div className="space-y-2 text-sm">
                            <Info label="Model" value={scan?.modelKey} />
                            <Info label="Anatomy" value={scan?.anatomy} />
                            <Info label="Threshold" value={scan?.threshold} />
                            <Info label="Smoothing" value={scan?.smooth} />
                            <Info label="Volume" value={scan?.volumeShape?.join(" × ")} />
                            <Info label="Vertices" value={scan?.vertexCount?.toLocaleString()} />
                            <Info label="Faces" value={scan?.faceCount?.toLocaleString()} />
                            <Info label="Status" value={scan?.status} />
                        </div>
                    </div>

                    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-4">
                        <h3 className="font-semibold text-gray-100">Metadata</h3>
                        <div className="space-y-2 text-sm">
                            <Info label="Doctor" value={scan?.doctor?.name} />
                            <Info label="AP File" value={scan?.apFileName} />
                            <Info label="Lateral File" value={scan?.latFileName || null} />
                            <Info label="Date" value={scan?.createdAt ? new Date(scan.createdAt).toLocaleString() : null} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function Info({ label, value }) {
    return (
        <div className="flex justify-between">
            <span className="text-gray-500">{label}</span>
            <span className="text-gray-200 font-medium">{value || "—"}</span>
        </div>
    );
}
