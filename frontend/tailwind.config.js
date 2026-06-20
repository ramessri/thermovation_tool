/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f9ff",
          500: "#0ea5e9",
          600: "#0284c7",
          900: "#0c4a6e",
        },
        coverage: {
          low:    "#ef4444",  // red
          medium: "#f59e0b",  // amber
          high:   "#22c55e",  // green
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
