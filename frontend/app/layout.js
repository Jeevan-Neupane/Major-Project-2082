import "./globals.css";

export const metadata = {
    title: "BoneView — X-Ray to 3D Reconstruction",
    description: "Hospital bone reconstruction platform",
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body className="bg-gray-950 text-gray-100 min-h-screen">{children}</body>
        </html>
    );
}
