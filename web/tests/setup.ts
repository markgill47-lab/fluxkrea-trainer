/**
 * Test environment setup.
 *
 * jsdom is missing several browser APIs the client and its dependencies
 * use. They are stubbed here so a component under test fails for its own
 * reasons rather than for the environment's.
 *
 * **These run at module scope, not in `beforeEach`.** uPlot calls
 * `matchMedia` while it is being imported, so a stub installed in a hook
 * arrives after the module graph has already thrown.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/preact";
import { afterEach, vi } from "vitest";

// uPlot reads the device pixel ratio at import time.
if (!globalThis.matchMedia) {
  globalThis.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof globalThis.matchMedia;
}

// The gallery and the monitor measure elements to decide how many rows to
// render. jsdom reports every element as 0x0, so a virtualizer renders
// nothing — which would make a list assertion fail for the wrong reason.
if (!("ResizeObserver" in globalThis)) {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
}

if (!("IntersectionObserver" in globalThis)) {
  globalThis.IntersectionObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
    takeRecords(): [] {
      return [];
    }
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds: number[] = [];
  } as unknown as typeof IntersectionObserver;
}

// The monitor opens an EventSource. Nothing here tests streaming, but a
// component that constructs one must not throw on mount.
if (!("EventSource" in globalThis)) {
  globalThis.EventSource = class {
    close(): void {}
    addEventListener(): void {}
    removeEventListener(): void {}
    readonly readyState = 0;
    onmessage: unknown = null;
    onerror: unknown = null;
  } as unknown as typeof EventSource;
}

if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = vi.fn() as unknown as typeof Element.prototype.scrollTo;
}

if (!HTMLCanvasElement.prototype.getContext) {
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => null,
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
});
