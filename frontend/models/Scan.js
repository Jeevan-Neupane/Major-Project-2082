import mongoose from "mongoose";

const ScanSchema = new mongoose.Schema({
    patient: { type: mongoose.Schema.Types.ObjectId, ref: "Patient", required: true },
    doctor: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },

    // What was uploaded
    modelKey: { type: String, required: true },
    anatomy: { type: String, default: "" },          // femur, hip, rib, etc.
    apFileName: { type: String, default: "" },
    latFileName: { type: String, default: "" },
    threshold: { type: Number, default: 0.5 },
    smooth: { type: Number, default: 1.0 },

    // Reconstruction output
    volumeShape: [Number],
    vertexCount: { type: Number, default: 0 },
    faceCount: { type: Number, default: 0 },
    meshFilePath: { type: String, default: "" },  // path to saved mesh JSON

    // Status tracking — like a real hospital RIS
    status: { type: String, enum: ["pending", "processing", "completed", "failed"], default: "completed" },
    diagnosis: { type: String, default: "" },           // doctor can annotate after viewing
    notes: { type: String, default: "" },
}, { timestamps: true });

export default mongoose.models.Scan || mongoose.model("Scan", ScanSchema);
