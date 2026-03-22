import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { createPatient, findPatientById, addScan } from "@/lib/db";
import fs from "fs/promises";
import path from "path";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const SCAN_DATA_DIR = path.join(process.cwd(), "scan_data");

/**
 * POST /api/predict
 * Body: FormData with ap_image, lat_image (optional), model_key,
 *       patient_name, patient_age, patient_gender, patient_phone, patient_notes,
 *       patient_id (if existing patient),
 *       threshold, smooth, output ("mesh" | "nifti")
 */
export async function POST(request) {
    const session = await getSession();
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await request.formData();
    const modelKey = formData.get("model_key");
    const output = formData.get("output") || "mesh";
    const apImage = formData.get("ap_image");
    const latImage = formData.get("lat_image");
    const threshold = formData.get("threshold") || "0.5";
    const smooth = formData.get("smooth") || "1.0";

    // Patient info
    const patientId = formData.get("patient_id");          // existing patient
    const patientName = formData.get("patient_name") || "Unknown";
    const patientAge = formData.get("patient_age");
    const patientGender = formData.get("patient_gender") || "";
    const patientPhone = formData.get("patient_phone") || "";
    const patientNotes = formData.get("patient_notes") || "";

    if (!modelKey || !apImage) {
        return NextResponse.json(
            { error: "model_key and ap_image are required" },
            { status: 400 }
        );
    }

    // Resolve or create patient
    let patient;
    if (patientId) {
        patient = await findPatientById(patientId);
        if (!patient) {
            return NextResponse.json({ error: "Patient not found" }, { status: 404 });
        }
    } else {
        patient = await createPatient({
            name: patientName,
            age: patientAge ? parseInt(patientAge, 10) : undefined,
            gender: patientGender,
            phone: patientPhone,
            referredBy: session.id,
            notes: patientNotes,
        });
    }

    // Build form data for FastAPI backend
    const backendForm = new FormData();
    backendForm.append("ap_image", apImage);
    if (latImage && latImage.size > 0) {
        backendForm.append("lat_image", latImage);
    }

    const endpoint =
        output === "mesh"
            ? `${API}/predict/${modelKey}/mesh?threshold=${threshold}&smooth=${smooth}`
            : `${API}/predict/${modelKey}?threshold=${threshold}`;

    // Derive anatomy from model key
    const anatomyMatch = modelKey.match(/(femur|hip|rib|verse19|verse20|ctpelvic)/);
    const anatomy = anatomyMatch ? anatomyMatch[1] : "";

    try {
        const backendRes = await fetch(endpoint, {
            method: "POST",
            body: backendForm,
        });

        if (!backendRes.ok) {
            const err = await backendRes.text();
            return NextResponse.json(
                { error: "Backend error", detail: err },
                { status: backendRes.status }
            );
        }

        if (output === "mesh") {
            const meshData = await backendRes.json();

            // Save mesh data to disk for later review
            await fs.mkdir(SCAN_DATA_DIR, { recursive: true });
            const meshFileName = `${Date.now()}_${modelKey}.json`;
            const meshFilePath = path.join(SCAN_DATA_DIR, meshFileName);
            await fs.writeFile(meshFilePath, JSON.stringify({
                vertices: meshData.vertices,
                faces: meshData.faces,
                normals: meshData.normals,
                vertex_count: meshData.vertex_count,
                face_count: meshData.face_count,
                volume_shape: meshData.volume_shape,
            }));

            // Save scan record with full tracking
            const scan = await addScan({
                patient: patient._id,
                doctor: session.id,
                modelKey,
                anatomy,
                apFileName: apImage.name || "ap_image",
                latFileName: latImage?.name || "",
                threshold: parseFloat(threshold),
                smooth: parseFloat(smooth),
                volumeShape: meshData.volume_shape || [],
                vertexCount: meshData.vertex_count || 0,
                faceCount: meshData.face_count || 0,
                meshFilePath: meshFileName,
                status: "completed",
            });

            return NextResponse.json({
                ...meshData,
                scanId: scan._id,
                patient: { id: patient._id, patientId: patient.patientId, name: patient.name },
            });
        } else {
            const buffer = await backendRes.arrayBuffer();

            await addScan({
                patient: patient._id,
                doctor: session.id,
                modelKey,
                anatomy,
                apFileName: apImage.name || "ap_image",
                latFileName: latImage?.name || "",
                threshold: parseFloat(threshold),
                smooth: parseFloat(smooth),
                status: "completed",
            });

            return new NextResponse(buffer, {
                headers: {
                    "Content-Type": "application/gzip",
                    "Content-Disposition": `attachment; filename="${modelKey}_output.nii.gz"`,
                },
            });
        }
    } catch (err) {
        return NextResponse.json(
            { error: "Backend unreachable", detail: err.message },
            { status: 502 }
        );
    }
}
