import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "LV_");
  const apiTarget = process.env.LV_WEB_API_TARGET || env.LV_WEB_API_TARGET || "http://127.0.0.1:8765";
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      proxy: {
        "/api": { target: apiTarget, changeOrigin: false },
      },
    },
  };
});
