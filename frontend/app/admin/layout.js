import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";

export const metadata = {
    title: "Admin — BoneView",
};

export default function AdminLayout({ children }) {
    return (
        <div className="min-h-screen bg-gray-950">
            <Navbar />
            <div className="flex">
                <Sidebar />
                <main className="flex-1 p-6 overflow-auto">{children}</main>
            </div>
        </div>
    );
}
