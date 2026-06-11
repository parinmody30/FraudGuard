import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      // Optional: proxy API in dev so VITE_API_BASE_URL can stay empty.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
