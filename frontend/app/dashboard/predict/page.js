"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";

const BoneViewer = dynamic(() => import("@/components/BoneViewer"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-[500px] bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center">
            <p className="text-gray-500 text-sm">Loading 3D viewer...</p>
        </div>
    ),
});

const ANATOMY_MAP = {
    femur: "Femur",
    hip: "Hip",
    rib: "Rib",
    verse19: "Vertebra (v19)",
    verse20: "Vertebra (v20)",
    ctpelvic: "CT Pelvic",
};

function StepBadge({ n, done }) {
    return (
        <span className={`w-6 h-6 rounded-full text-xs font-bold flex items-center justify-center flex-shrink-0 transition-colors ${done ? "bg-green-800 text-green-300" : "bg-primary-900/40 text-primary-400"}`}>
            {done ? (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
            ) : n}
        </span>
    );
}

function Stat({ label, value }) {
    return (
        <div>
            <p className="text-xs text-gray-500">{label}</p>
            <p className="text-gray-200 font-medium text-sm truncate">{value || ""}</p>
        </div>
    );
}

export default function PredictPage() {
    const [patientMode, setPatientMode] = useState("new");       // "new" | "existing"
    const [patientSearch, setPatientSearch] = useState("");
    const [patientResults, setPatientResults] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState(null); // existing patient picked from search
    const [patientName, setPatientName] = useState("");
    const [patientAge, setPatientAge] = useState("");
    const [patientGender, setPatientGender] = useState("");
    const [patientPhone, setPatientPhone] = useState("");
    const [patientNotes, setPatientNotes] = useState("");
    const [savingPatient, setSavingPatient] = useState(false);
    const [savedPatient, setSavedPatient] = useState(null);      // confirmed/saved patient :’ unlocks step 2
    const [patientError, setPatientError] = useState("");

    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState("");
    const [apFile, setApFile] = useState(null);
    const [latFile, setLatFile] = useState(null);
    const [apPreview, setApPreview] = useState(null);
    const [latPreview, setLatPreview] = useState(null);
    const [threshold, setThreshold] = useState(0.5);
    const [smooth, setSmooth] = useState(1.0);

    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState("");
    const [meshData, setMeshData] = useState(null);
    const [scanResult, setScanResult] = useState(null);
    const [error, setError] = useState("");

    // Fetch available models
    useEffect(() => {
        fetch("/api/models")
            .then((r) => r.json())
            .then((d) => {
                if (Array.isArray(d)) {
                    const avail = d.filter((m) => m.weights_exist);
                    setModels(avail);
                    if (avail.length > 0) setSelectedModel(avail[0].model_key);
                }
            })
            .catch(() => { });
    }, []);

    // Search patients (debounced)
    useEffect(() => {
        if (patientMode !== "existing" || patientSearch.length < 2) {
            setPatientResults([]);
            return;
        }
        const timeout = setTimeout(() => {
            fetch(`/api/patients?q=${encodeURIComponent(patientSearch)}`)
                .then((r) => r.json())
                .then((d) => setPatientResults(Array.isArray(d) ? d : []))
                .catch(() => { });
        }, 300);
        return () => clearTimeout(timeout);
    }, [patientSearch, patientMode]);

    const handleFileChange = useCallback((setter, previewSetter) => (e) => {
        const file = e.target.files?.[0];
        if (file) {
            setter(file);
            const reader = new FileReader();
            reader.onload = (ev) => previewSetter(ev.target.result);
            reader.readAsDataURL(file);
        }
    }, []);

    async function handleSavePatient(e) {
        e.preventDefault();
        setPatientError("");

        if (patientMode === "existing") {
            if (!selectedPatient) {
                setPatientError("Please search and select an existing patient.");
                return;
            }
            setSavedPatient(selectedPatient);
            return;
        }

        if (!patientName.trim()) {
            setPatientError("Patient name is required.");
            return;
        }

        setSavingPatient(true);
        try {
            const res = await fetch("/api/patients", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: patientName.trim(),
                    age: patientAge || undefined,
                    gender: patientGender || undefined,
                    phone: patientPhone || undefined,
                    notes: patientNotes || undefined,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Failed to save patient");
            setSavedPatient(data);
        } catch (err) {
            setPatientError(err.message);
        } finally {
            setSavingPatient(false);
        }
    }

    function handleChangePatient() {
        setSavedPatient(null);
        setMeshData(null);
        setScanResult(null);
        setError("");
    }

    // :€:€ Step 2: Generate 3D :€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€
    async function handlePredict(e) {
        e.preventDefault();
        if (!savedPatient) {
            setError("Please register or confirm the patient first (Step 1).");
            return;
        }
        if (!selectedModel) {
            setError("Please select an AI model.");
            return;
        }
        if (!apFile) {
            setError("AP (frontal) X-ray image is required.");
            return;
        }

        setError("");
        setMeshData(null);
        setScanResult(null);
        setLoading(true);
        setProgress("Uploading X-rays...");

        const form = new FormData();
        form.append("model_key", selectedModel);
        form.append("output", "mesh");
        form.append("ap_image", apFile);
        if (latFile) form.append("lat_image", latFile);
        form.append("threshold", threshold.toString());
        form.append("smooth", smooth.toString());
        form.append("patient_id", savedPatient._id);

        try {
            setProgress("Running AI model  reconstructing 3D bone...");
            const res = await fetch("/api/predict", { method: "POST", body: form });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || data.detail || "Prediction failed");
            }

            setProgress("Building 3D mesh...");
            const data = await res.json();
            setMeshData(data);
            setScanResult(data);
            setProgress("");
        } catch (err) {
            setError(err.message);
            setProgress("");
        } finally {
            setLoading(false);
        }
    }

    const anatomy = Object.keys(ANATOMY_MAP).find((k) => selectedModel.includes(k));
    const step1Done = !!savedPatient;

    return (
        <div className="max-w-7xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-100 mb-1">New Scan Order</h1>
            <p className="text-gray-500 text-sm mb-6">
                Register the patient first, then upload X-rays and generate the 3D reconstruction.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* :€:€ Left Column :€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€ */}
                <div className="lg:col-span-2 space-y-4">

                    {/* :€:€ STEP 1: Patient :€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€:€ */}
                    <div className={`bg-gray-900 rounded-xl border p-5 space-y-4 transition-colors ${step1Done ? "border-green-800/60" : "border-gray-800"}`}>
                        <div className="flex items-center justify-between">
                            <h2 className="font-semibold text-gray-100 flex items-center gap-2">
                                <StepBadge n="1" done={step1Done} />
                                Patient Registration
                            </h2>
                            {step1Done ? (
                                <button type="button" onClick={handleChangePatient}
                                    className="text-xs text-primary-400 hover:text-primary-300 font-medium">
                                    Change
                                </button>
                            ) : (
                                <div className="flex bg-gray-800 rounded-lg p-0.5 text-xs">
                                    <button type="button"
                                        onClick={() => { setPatientMode("new"); setSelectedPatient(null); }}
                                        className={`px-3 py-1.5 rounded-md font-medium transition-colors ${patientMode === "new" ? "bg-gray-700 shadow text-gray-100" : "text-gray-500"}`}>
                                        New
                                    </button>
                                    <button type="button" onClick={() => setPatientMode("existing")}
                                        className={`px-3 py-1.5 rounded-md font-medium transition-colors ${patientMode === "existing" ? "bg-gray-700 shadow text-gray-100" : "text-gray-500"}`}>
                                        Existing
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Confirmed patient banner */}
                        {step1Done ? (
                            <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-3 flex items-center gap-3">
                                <div className="w-9 h-9 rounded-full bg-green-800/40 flex items-center justify-center text-green-300 font-bold text-sm flex-shrink-0">
                                    {savedPatient.name?.[0]?.toUpperCase()}
                                </div>
                                <div className="min-w-0">
                                    <p className="font-semibold text-green-200 text-sm truncate">{savedPatient.name}</p>
                                    <p className="text-green-600 text-xs">{savedPatient.patientId} &middot; {savedPatient.age ? `${savedPatient.age}y` : ""} &middot; {savedPatient.gender || ""}</p>
                                </div>
                                <svg className="w-5 h-5 text-green-500 flex-shrink-0 ml-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                        ) : (
                            <form onSubmit={handleSavePatient} className="space-y-3">
                                {patientMode === "existing" ? (
                                    <div className="space-y-2">
                                        <input
                                            type="text"
                                            placeholder="Search by name or Patient ID..."
                                            value={patientSearch}
                                            onChange={(e) => { setPatientSearch(e.target.value); setSelectedPatient(null); }}
                                            className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600"
                                        />
                                        {selectedPatient && (
                                            <div className="bg-primary-900/30 border border-primary-800 rounded-lg p-3 text-sm flex justify-between items-center">
                                                <div>
                                                    <p className="font-medium text-primary-300">{selectedPatient.name}</p>
                                                    <p className="text-primary-500 text-xs">{selectedPatient.patientId} &middot; {selectedPatient.age ? `${selectedPatient.age}y` : ""} {selectedPatient.gender}</p>
                                                </div>
                                                <button type="button" onClick={() => { setSelectedPatient(null); setPatientSearch(""); }}
                                                    className="text-primary-400 text-xs hover:text-primary-300">Clear</button>
                                            </div>
                                        )}
                                        {!selectedPatient && patientResults.length > 0 && (
                                            <div className="border border-gray-700 rounded-lg overflow-hidden max-h-40 overflow-y-auto">
                                                {patientResults.map((p) => (
                                                    <button key={p._id} type="button"
                                                        onClick={() => { setSelectedPatient(p); setPatientResults([]); setPatientSearch(p.name); }}
                                                        className="w-full text-left px-3 py-2 hover:bg-gray-800 border-b border-gray-700 last:border-0 text-sm">
                                                        <span className="font-medium text-gray-200">{p.name}</span>
                                                        <span className="text-gray-500 ml-2 text-xs">{p.patientId} &middot; {p.age ? `${p.age}y` : ""} {p.gender}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="col-span-2">
                                            <label className="block text-xs font-medium text-gray-400 mb-1">Full Name <span className="text-red-400">*</span></label>
                                            <input type="text" value={patientName} onChange={(e) => setPatientName(e.target.value)}
                                                placeholder="e.g. John Doe"
                                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600" />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-gray-400 mb-1">Age</label>
                                            <input type="number" value={patientAge} onChange={(e) => setPatientAge(e.target.value)}
                                                min="0" max="150" placeholder="45"
                                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600" />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-gray-400 mb-1">Gender</label>
                                            <select value={patientGender} onChange={(e) => setPatientGender(e.target.value)}
                                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none">
                                                <option value=""></option>
                                                <option value="Male">Male</option>
                                                <option value="Female">Female</option>
                                                <option value="Other">Other</option>
                                            </select>
                                        </div>
                                        <div className="col-span-2">
                                            <label className="block text-xs font-medium text-gray-400 mb-1">Phone</label>
                                            <input type="tel" value={patientPhone} onChange={(e) => setPatientPhone(e.target.value)}
                                                placeholder="+91 98765 43210"
                                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600" />
                                        </div>
                                        <div className="col-span-2">
                                            <label className="block text-xs font-medium text-gray-400 mb-1">Clinical Notes</label>
                                            <textarea value={patientNotes} onChange={(e) => setPatientNotes(e.target.value)}
                                                rows={2} placeholder="e.g. Left hip pain since 2 weeks"
                                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none placeholder-gray-600 resize-none" />
                                        </div>
                                    </div>
                                )}

                                {patientError && (
                                    <p className="text-red-400 text-xs bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">{patientError}</p>
                                )}

                                <button type="submit" disabled={savingPatient}
                                    className="w-full py-2.5 px-4 bg-primary-600 text-white rounded-xl text-sm font-medium hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                                    {savingPatient ? (
                                        <>
                                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" /></svg>
                                            Saving...
                                        </>
                                    ) : patientMode === "existing" ? (
                                        "Confirm Patient"
                                    ) : (
                                        "Save Patient & Continue"
                                    )}
                                </button>
                            </form>
                        )}
                    </div>

                    <form onSubmit={handlePredict}
                        className={`bg-gray-900 rounded-xl border p-5 space-y-4 transition-all ${step1Done ? "border-gray-800" : "border-gray-800/50 opacity-50 pointer-events-none select-none"}`}
                    >
                        <h2 className="font-semibold text-gray-100 flex items-center gap-2">
                            <StepBadge n="2" done={!!scanResult} />
                            X-Ray Upload & Reconstruction
                            {!step1Done && <span className="ml-auto text-xs text-gray-600 font-normal">(complete Step 1 first)</span>}
                        </h2>

                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">AI Model</label>
                            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-100 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none">
                                {models.length === 0 && <option value="">No models available is the AI backend running?</option>}
                                {models.map((m) => (
                                    <option key={m.model_key} value={m.model_key}>{m.model_key} {m.description}</option>
                                ))}
                            </select>
                            {anatomy && <p className="mt-1 text-xs text-gray-500">Anatomy: {ANATOMY_MAP[anatomy]}</p>}
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1">AP X-Ray <span className="text-red-400">*</span></label>
                                <input type="file" accept="image/*" onChange={handleFileChange(setApFile, setApPreview)}
                                    className="w-full text-xs text-gray-400 file:mr-2 file:px-3 file:py-1.5 file:rounded-lg file:border-0 file:bg-primary-900/40 file:text-primary-400 file:font-medium file:cursor-pointer hover:file:bg-primary-900/60" />
                                {apPreview && <img src={apPreview} alt="AP" className="mt-2 rounded-lg border border-gray-700 max-h-28 object-contain bg-black" />}
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1">Lateral X-Ray</label>
                                <input type="file" accept="image/*" onChange={handleFileChange(setLatFile, setLatPreview)}
                                    className="w-full text-xs text-gray-400 file:mr-2 file:px-3 file:py-1.5 file:rounded-lg file:border-0 file:bg-gray-800 file:text-gray-300 file:font-medium file:cursor-pointer hover:file:bg-gray-700" />
                                {latPreview && <img src={latPreview} alt="Lateral" className="mt-2 rounded-lg border border-gray-700 max-h-28 object-contain bg-black" />}
                            </div>
                        </div>

                        <details className="text-sm">
                            <summary className="cursor-pointer text-gray-400 font-medium text-xs">Advanced Settings</summary>
                            <div className="mt-3 space-y-3">
                                <div>
                                    <label className="flex items-center justify-between text-gray-400 mb-1 text-xs">
                                        Threshold <span className="text-gray-500">{threshold}</span>
                                    </label>
                                    <input type="range" min="0.1" max="0.9" step="0.05" value={threshold}
                                        onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full accent-primary-600" />
                                </div>
                                <div>
                                    <label className="flex items-center justify-between text-gray-400 mb-1 text-xs">
                                        Smoothing <span className="text-gray-500">{smooth}</span>
                                    </label>
                                    <input type="range" min="0" max="3" step="0.25" value={smooth}
                                        onChange={(e) => setSmooth(parseFloat(e.target.value))} className="w-full accent-primary-600" />
                                </div>
                            </div>
                        </details>

                        {error && (
                            <div className="bg-red-900/30 text-red-400 text-xs px-3 py-2.5 rounded-lg border border-red-800">{error}</div>
                        )}

                        <button type="submit"
                            disabled={loading || !apFile || !selectedModel || !step1Done}
                            className="w-full py-3 px-4 bg-primary-600 text-white rounded-xl font-medium hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                                    </svg>
                                    {progress || "Processing..."}
                                </span>
                            ) : "Generate 3D Reconstruction"}
                        </button>
                    </form>
                </div>

                <div className="lg:col-span-3 space-y-4">
                    <BoneViewer meshData={meshData} />

                    {scanResult && (
                        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-3">
                            <div className="flex items-center justify-between">
                                <h3 className="font-semibold text-gray-100">Scan Report</h3>
                                <div className="flex items-center gap-2">
                                    {scanResult.scanId && (
                                        <a href={`/dashboard/history/${scanResult.scanId}`}
                                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 text-xs font-medium hover:bg-gray-700 transition-colors">
                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
                                            </svg>
                                            View in History
                                        </a>
                                    )}
                                    {meshData && (
                                        <button
                                            onClick={() => {
                                                const lines = [];
                                                for (const v of meshData.vertices) lines.push(`v ${v[0]} ${v[1]} ${v[2]}`);
                                                if (meshData.normals) for (const n of meshData.normals) lines.push(`vn ${n[0]} ${n[1]} ${n[2]}`);
                                                for (const f of meshData.faces) lines.push(`f ${f[0] + 1} ${f[1] + 1} ${f[2] + 1}`);
                                                const blob = new Blob([lines.join("\n")], { type: "text/plain" });
                                                const url = URL.createObjectURL(blob);
                                                const a = document.createElement("a");
                                                a.href = url; a.download = `${selectedModel}_reconstruction.obj`; a.click();
                                                URL.revokeObjectURL(url);
                                            }}
                                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary-600 text-white text-xs font-medium hover:bg-primary-700 transition-colors">
                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                            </svg>
                                            Download OBJ
                                        </button>
                                    )}
                                </div>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                                <Stat label="Patient" value={scanResult.patient?.name} />
                                <Stat label="Patient ID" value={scanResult.patient?.patientId} />
                                <Stat label="Model" value={scanResult.model_key} />
                                <Stat label="Volume" value={scanResult.volume_shape?.join(" Ã— ")} />
                                <Stat label="Vertices" value={scanResult.vertex_count?.toLocaleString()} />
                                <Stat label="Faces" value={scanResult.face_count?.toLocaleString()} />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
