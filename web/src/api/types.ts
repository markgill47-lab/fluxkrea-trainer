/**
 * The shapes the daemon actually returns.
 *
 * Hand-written on purpose. Doc 02 forbids coupling the build to the API,
 * so there is no codegen step pointed at a running server — these types
 * are a claim about the contract, and the API contract tests on the
 * Python side are what keep the claim true.
 */

export interface Health {
  status: string;
  version: string;
  node: string;
  uptime: number;
  queue_depth: number;
  tasks_active: number;
}

export interface Gpu {
  index: number;
  name: string;
  capability: string;
  vram_total: number;
  vram_free: number;
}

export interface NodeInfo {
  name: string;
  version: string;
  hostname: string;
  os: string;
  os_release: string;
  machine: string;
  python: string;
  opencv: string;
  torch: string | null;
  cuda: string | null;
  driver: string | null;
  detectors: Record<string, boolean>;
  backends: Record<string, { ready: boolean; models: string[] }>;
  gpus: Gpu[];
  disk_free: { total: number; free: number } | null;
}

export interface Dataset {
  id: string;
  path: string;
  name: string;
  exists: boolean;
}

export interface Item {
  stem: string;
  filename: string;
  caption: string | null;
  has_caption: boolean;
  has_mask: boolean;
  quality: string | null;
  boxes: number;
  reviewed: boolean;
  /** Cached in metadata.json, keyed by the same token as the thumbnail. */
  width: number | null;
  height: number | null;
  /** Cache token; changes when the file does. Makes the thumb URL immutable. */
  token: string;
}

export interface ReviewProgress {
  total: number;
  reviewed: number;
  complete: boolean;
  /** Images with no boxes at all. Doc 04: this is where misses hide. */
  empty: string[];
  /** Images detection has never run against. */
  undetected: string[];
  summary: string;
}

export interface ItemsResponse {
  items: Item[];
  review: ReviewProgress;
}

/** Box source. `manual` boxes are drawn by a human and survive re-detection. */
export type BoxSource = "manual" | "yunet" | "unknown" | string;

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  src: BoxSource;
  conf?: number | null;
}

export interface BoxesResponse {
  stem: string;
  filename?: string;
  boxes: Box[];
  reviewed: boolean;
}

export type Severity = "error" | "warning" | "info";

export interface Problem {
  kind: string;
  severity: Severity;
  message: string;
  stem: string | null;
  path: string | null;
}

export interface ValidationReport {
  root: string;
  items: number;
  ok: boolean;
  counts: Record<string, number>;
  problems: Problem[];
}

export type TaskStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Task {
  id: string;
  kind: string;
  dataset: string | null;
  status: TaskStatus;
  created: number;
  started: number | null;
  finished: number | null;
  detail: { operation?: string; progress?: { step: number; total: number } };
  error: string;
  events: number;
  result?: Record<string, unknown>;
}

/** The event vocabulary, identical to the core dataclasses (doc 02). */
export type StreamEvent =
  | { kind: "progress"; index: number; step: number; total: number; message: string }
  | { kind: "log"; index: number; line: string; level: "debug" | "info" | "warning" | "error" }
  | { kind: "loss"; index: number; step: number; value: number; image_id: string | null }
  | { kind: "finished"; index: number; ok: boolean; detail: string };

export interface ModelInfo {
  id: string;
  arch: string;
  label: string;
  network_dim: number;
  network_alpha: number;
  low_vram: boolean;
  guidance_scale: number;
  text_encoder: string;
  notes: string;
}
