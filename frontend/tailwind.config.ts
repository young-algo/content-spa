/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "selector",
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        // Workbench OKLCH palette. Registered as real Tailwind colors (not just
        // @layer utilities) so /opacity modifiers — bg-cobalt/90, bg-cobalt-light/30,
        // border-cobalt/25, text-ink-muted/70 — actually generate CSS. The
        // <alpha-value> placeholder is what lets Tailwind inject alpha into oklch().
        cobalt: "oklch(52% 0.19 250 / <alpha-value>)",
        "cobalt-light": "oklch(95% 0.02 250 / <alpha-value>)",
        cyan: "oklch(64% 0.15 195 / <alpha-value>)",
        "cyan-light": "oklch(95% 0.02 195 / <alpha-value>)",
        ink: {
          DEFAULT: "oklch(16% 0.02 240 / <alpha-value>)",
          muted: "oklch(48% 0.015 240 / <alpha-value>)",
          border: "oklch(91% 0.006 240 / <alpha-value>)",
        },
        paper: "oklch(98.5% 0.003 240 / <alpha-value>)",
        pure: "oklch(100% 0 0 / <alpha-value>)",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
};
