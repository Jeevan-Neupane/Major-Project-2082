/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./app/**/*.{js,jsx}",
        "./components/**/*.{js,jsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    50: "#eff6ff",
                    100: "#dbeafe",
                    200: "#bfdbfe",
                    300: "#93c5fd",
                    400: "#60a5fa",
                    500: "#3b82f6",
                    600: "#2563eb",
                    700: "#1d4ed8",
                    800: "#1e40af",
                    900: "#1e3a8a",
                },
                // Ivory bone-tone accent — the subject of the platform itself.
                bone: {
                    50: "#F8F6F0",
                    100: "#EFEAE0",
                    200: "#E2DAC9",
                    300: "#CFC4AC",
                    400: "#B3A687",
                    500: "#96886B",
                    600: "#786B51",
                    700: "#5E5440",
                    800: "#413A2C",
                    900: "#2A251D",
                },
            },
            fontFamily: {
                clinical: ['"IBM Plex Sans"', "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
                readout: ['"IBM Plex Mono"', "ui-monospace", "Consolas", "monospace"],
            },
        },
    },
    plugins: [],
};
