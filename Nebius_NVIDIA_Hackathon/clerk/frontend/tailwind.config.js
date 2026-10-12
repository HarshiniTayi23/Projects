/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["Fraunces", "ui-serif", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        ink: { DEFAULT: "#14161a", soft: "#3b3f47", mute: "#6b7280" },
        paper: { DEFAULT: "#f7f5f0", card: "#ffffff", line: "#e6e2d8" },
        nv: { DEFAULT: "#76b900", dark: "#4f7d00" },
        upheld: { DEFAULT: "#0f7b4b", bg: "#e8f5ee" },
        dismissed: { DEFAULT: "#8a8f98", bg: "#f1f1ef" },
        pending: { DEFAULT: "#b7791f", bg: "#fdf5e6" },
      },
      keyframes: {
        rise: { "0%": { opacity: "0", transform: "translateY(6px)" }, "100%": { opacity: "1", transform: "none" } },
      },
      animation: { rise: "rise 300ms ease-out both" },
    },
  },
  plugins: [],
};
