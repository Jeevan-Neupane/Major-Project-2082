import mongoose from "mongoose";

const PatientSchema = new mongoose.Schema({
    patientId: { type: String, unique: true },                          // auto-generated hospital ID
    name: { type: String, required: true, trim: true },
    age: { type: Number, min: 0, max: 150 },
    gender: { type: String, enum: ["Male", "Female", "Other", ""], default: "" },
    phone: { type: String, default: "" },
    referredBy: { type: mongoose.Schema.Types.ObjectId, ref: "User" },  // doctor who registered this patient
    notes: { type: String, default: "" },                           // clinical notes
}, { timestamps: true });

// Auto-generate a readable hospital ID like "PT-00001"
PatientSchema.pre("save", async function (next) {
    if (!this.patientId) {
        const count = await mongoose.models.Patient.countDocuments();
        this.patientId = `PT-${String(count + 1).padStart(5, "0")}`;
    }
    next();
});

export default mongoose.models.Patient || mongoose.model("Patient", PatientSchema);
