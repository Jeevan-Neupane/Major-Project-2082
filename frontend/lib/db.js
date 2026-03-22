/**
 * MongoDB-backed database helpers for users, patients, and scans.
 * On first call the default admin account is seeded automatically.
 */
import bcrypt from "bcryptjs";
import dbConnect from "@/lib/mongodb";
import User from "@/models/User";
import Patient from "@/models/Patient";
import Scan from "@/models/Scan";

// ── seed admin (runs once) ─────────────────────────────────────────
let _seeded = false;
export async function seedAdmin() {
    await dbConnect();
    if (_seeded) return;
    const existing = await User.findOne({ username: "admin" }).lean();
    if (!existing) {
        await User.create({
            username: "admin",
            name: "Administrator",
            role: "admin",
            passwordHash: bcrypt.hashSync("admin123", 10),
        });
    }
    _seeded = true;
}

// ── users ──────────────────────────────────────────────────────────
export async function getUsers() {
    await seedAdmin();
    return User.find({ isActive: true }).lean();
}

export async function findUser(username) {
    await seedAdmin();
    return User.findOne({ username, isActive: true }).lean();
}

export async function findUserById(id) {
    await seedAdmin();
    return User.findById(id).lean();
}

export function verifyPassword(plain, hash) {
    return bcrypt.compareSync(plain, hash);
}

export async function addUser({ username, name, role = "doctor", password, speciality = "", phone = "" }) {
    await dbConnect();
    const exists = await User.findOne({ username }).lean();
    if (exists) return { error: "Username already exists" };

    const user = await User.create({
        username,
        name,
        role,
        passwordHash: bcrypt.hashSync(password, 10),
        speciality,
        phone,
    });
    return { user: user.toObject() };
}

export async function deleteUser(id) {
    await dbConnect();
    const user = await User.findById(id);
    if (!user) return { error: "User not found" };
    if (user.role === "admin") return { error: "Cannot delete admin" };
    user.isActive = false;
    await user.save();
    return { deleted: true };
}

// ── patients ───────────────────────────────────────────────────────
export async function createPatient({ name, age, gender, phone, referredBy, notes }) {
    await dbConnect();
    const patient = await Patient.create({ name, age, gender, phone, referredBy, notes });
    return patient.toObject();
}

export async function getPatients(doctorId = null) {
    await dbConnect();
    const filter = doctorId ? { referredBy: doctorId } : {};
    return Patient.find(filter).sort({ createdAt: -1 }).lean();
}

export async function findPatientById(id) {
    await dbConnect();
    return Patient.findById(id).lean();
}

export async function searchPatients(query) {
    await dbConnect();
    return Patient.find({
        $or: [
            { name: { $regex: query, $options: "i" } },
            { patientId: { $regex: query, $options: "i" } },
        ],
    })
        .sort({ createdAt: -1 })
        .limit(20)
        .lean();
}

// ── scans (replaces old "results") ─────────────────────────────────
export async function addScan(data) {
    await dbConnect();
    const scan = await Scan.create(data);
    return scan.toObject();
}

export async function getScans(doctorId = null) {
    await dbConnect();
    const filter = doctorId ? { doctor: doctorId } : {};
    return Scan.find(filter)
        .populate("patient", "patientId name age gender")
        .populate("doctor", "name username")
        .sort({ createdAt: -1 })
        .lean();
}

export async function getScanById(scanId) {
    await dbConnect();
    return Scan.findById(scanId)
        .populate("patient", "patientId name age gender phone notes")
        .populate("doctor", "name username speciality")
        .lean();
}

export async function updateScanDiagnosis(scanId, { diagnosis, notes }) {
    await dbConnect();
    return Scan.findByIdAndUpdate(scanId, { diagnosis, notes }, { new: true }).lean();
}

export async function getDashboardStats(doctorId = null) {
    await dbConnect();
    const scanFilter = doctorId ? { doctor: doctorId } : {};
    const [totalScans, totalPatients, recentScans] = await Promise.all([
        Scan.countDocuments(scanFilter),
        doctorId ? Patient.countDocuments({ referredBy: doctorId }) : Patient.countDocuments(),
        Scan.find(scanFilter)
            .populate("patient", "patientId name")
            .populate("doctor", "name")
            .sort({ createdAt: -1 })
            .limit(5)
            .lean(),
    ]);
    return { totalScans, totalPatients, recentScans };
}
