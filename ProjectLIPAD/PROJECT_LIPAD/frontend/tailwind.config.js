/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        newsprint: "#F9F9F7",
        ink: "#111111",
        rule: "#E5E5E0",
        accent: "#CC0000",
        muted: "#737373",
      },
      fontFamily: {
        serif: ['"Times New Roman"', "Georgia", "serif"],
        sans: ["Inter", '"Helvetica Neue"', "sans-serif"],
        mono: ['"JetBrains Mono"', '"Courier New"', "monospace"],
      },
      borderRadius: {
        DEFAULT: "0",
        none: "0",
        sm: "0",
        md: "0",
        lg: "0",
        xl: "0",
        "2xl": "0",
        "3xl": "0",
        full: "0",
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        fadeIn: "fadeIn 0.35s ease-out",
      },
    },
  },
  plugins: [],
};
