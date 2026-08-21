import { fileURLToPath, URL } from "node:url";
import preact from "@preact/preset-vite";
import { defineConfig } from "vitest/config";

/**
 * Component tests for the client.
 *
 * These exist because of a specific failure: the training form kept its
 * settings in local state, so switching to the monitor and back unmounted
 * it and silently reset every field — the dataset falling back to the
 * first registered one. A run then trained the wrong images and said
 * nothing. No Python test could have caught it; it is a mount-lifecycle
 * bug that only exists once the component is on a page.
 *
 * So the bar for what goes here is: **behaviour that only appears when a
 * component is mounted, unmounted, re-rendered, or driven by a person.**
 * Not styling, not layout, not "does it render" — those are what the
 * browser pass is for. Lifecycle, state that must survive something, and
 * the exact shape of what gets sent to the API.
 *
 * A separate config from `vite.config.ts` rather than a `test` key in it:
 * the build config carries a dev-server proxy and manual chunking that
 * have nothing to do with tests, and the two drift apart more quietly when
 * they share a file.
 */
export default defineConfig({
  plugins: [preact()],
  resolve: {
    alias: { "~": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    // Fail rather than hang: a component waiting on a fetch that never
    // resolves should be a red test in seconds, not a stalled suite.
    testTimeout: 5000,
    restoreMocks: true,
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      reporter: ["text-summary"],
    },
  },
});
