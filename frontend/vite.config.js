import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.js",
  },
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target:
          loadEnv(mode, process.cwd(), "").PROTOGEN_API_PROXY ||
          "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
}));
