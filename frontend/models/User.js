import mongoose from "mongoose";

const UserSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true, trim: true, lowercase: true },
    name: { type: String, required: true, trim: true },
    role: { type: String, enum: ["admin", "doctor"], default: "doctor" },
    passwordHash: { type: String, required: true },
    speciality: { type: String, default: "" },       // e.g. Orthopaedics, Radiology
    phone: { type: String, default: "" },
    isActive: { type: Boolean, default: true },
}, { timestamps: true });

export default mongoose.models.User || mongoose.model("User", UserSchema);
