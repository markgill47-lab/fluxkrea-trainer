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
  /** Which captioners could be *built*, not whether they answer. */
  captioners: Record<string, boolean>;
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

// --------------------------------------------------------------------------
// training
// --------------------------------------------------------------------------

export type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface RunSpec {
  model: string;
  dataset: string;
  name: string;
  output: string;
  device: number;
  steps: number;
  batch_size: number;
  learning_rate: number;
  network_dim: number;
  network_alpha: number;
  resolution: number;
  mask_path: string;
  mask_min_value: number;
  sample_every: number;
  save_every: number;
  seed: number | null;
  extra: Record<string, unknown>;
}

export interface Job {
  id: string;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  error: string;
  device: number;
  config_path: string;
  progress: { step: number; total: number };
  events: number;
  spec: RunSpec;
}

export interface Trend {
  status: "improving" | "stable" | "degrading" | "converged" | "unknown";
  slope: number | null;
  window: number;
}

export interface Outlier {
  image_id: string;
  mean: number;
  severity: number;
  samples: number;
  /** Assigned client-side for drawing; the analytics identify which image,
   *  not when. */
  step?: number;
}

export interface LossPayload {
  id: string;
  points: { step: number; value: number }[];
  ema: { step: number; value: number }[];
  ema_window: number;
  count: number;
  decimated: boolean;
  latest: number | null;
  latest_ema: number | null;
  trend: Trend;
  outliers: Outlier[];
}

export type LossSeries = LossPayload;

export interface SampleImage {
  name: string;
  step: number;
  mtime: number;
  url: string;
}

// --------------------------------------------------------------------------
// settings
// --------------------------------------------------------------------------

export interface CaptionerConfig {
  provider: string;
  ollama_url: string;
  ollama_model: string;
  joycaption_model: string;
  joycaption_quantize: boolean;
  claude_model: string;
  prompt: string;
  prefix: string;
  max_tokens: number;
  timeout: number;
}

export interface MaskConfig {
  detector: string;
  confidence: number;
  expand: number;
  expand_up: number;
  feather: number;
  min_value: number;
  nms: number;
  require_review: boolean;
  write_previews: boolean;
}

/**
 * The whole config, as `GET /config` returns it. `read_only` names the
 * settings the API will refuse to write, so the UI can say *why* a field
 * is locked rather than discovering it through a 403.
 */
export interface ConfigPayload {
  dataset: Record<string, unknown>;
  mask: MaskConfig;
  captioner: CaptionerConfig;
  daemon: Record<string, unknown>;
  backends: Record<string, unknown>;
  log_level: string;
  source: string | null;
  read_only: string[];
  /** Present only on the response to a write. */
  changed?: string[];
  written?: string;
  restart_required?: string[];
}

export interface CaptionerInfo {
  name: string;
  label: string;
  available: boolean;
}

export interface CaptionerProbe {
  ok: boolean;
  message: string;
  provider?: string;
  /** Ollama only: what is actually pulled on the node. */
  models?: string[];
}

export interface SecretInfo {
  name: string;
  found: boolean;
  env: string[];
}
