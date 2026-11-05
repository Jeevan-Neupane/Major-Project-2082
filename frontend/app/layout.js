import "./globals.css";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";

const clinical = IBM_Plex_Sans({
    subsets: ["latin"],
    weight: ["400", "500", "600"],
    variable: "--font-clinical",
    display: "swap",
});

const plexReadout = IBM_Plex_Mono({
    subsets: ["latin"],
    weight: ["400", "500"],
    variable: "--font-readout",
    display: "swap",
});

export const metadata = {
    title: "BoneView — X-Ray to 3D Reconstruction",
    description: "Hospital bone reconstruction platform",
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body
                className={`${clinical.variable} ${plexReadout.variable} bg-gray-950 text-gray-100 font-clinical min-h-screen`}
            >
                {children}
            </body>
        </html>
    );
}
