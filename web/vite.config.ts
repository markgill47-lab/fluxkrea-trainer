import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

/**
 * The client is static assets served by the daemon (doc 02). Nothing here
 * may couple the build to the API: no codegen from a running server, no
 * environment baked in at build time. The client discovers everything it
 * needs at runtime from `/api/v1`.
 *
 * `base: "./"` so the bundle works wherever the daemon mounts it, which
 * over an SSH tunnel is not always the origin root.
 */
export default defineConfig({
  base: "./",
  plugins: [preact()],
  resolve: {
    // Mirrors the `~/*` path in tsconfig, so an import reads the same to
    // the type-checker and the bundler.
    alias: { "~": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    // The daemon serves this directory. Kept inside web/ so the Python
    // package layout is untouched.
    outDir: "dist",
    emptyOutDir: true,
    // Doc 10's budget is 500KB compressed for the whole client. Warn well
    // before that, in raw bytes, so it is noticed while there is room.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        // uPlot and the virtualizer are large enough to be worth caching
        // separately from application code that changes every commit.
        // uPlot joins this when the training monitor lands; splitting it
        // out now just produces an empty chunk and a warning that trains
        // people to ignore warnings.
        manualChunks: {
          vendor: ["preact", "@tanstack/virtual-core"],
        },
      },
    },
  },
  server: {
    port: 5173,
    // In development the client runs on Vite and the daemon on 8471.
    // Proxying keeps the client's fetch paths identical in both modes, so
    // there is no "am I in dev" branch anywhere in the app code.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8471",
        changeOrigin: true,
        // SSE must not be buffered, and must not time out on a quiet run.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache, no-transform";
            }
          });
        },
      },
    },
  },
});
