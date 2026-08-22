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
  /** The daemon is running code that has since been edited on disk. */
  stale?: boolean;
  /** How many projects exist, so the shell knows whether to offer a picker
   *  or go straight to "create one". */
  projects?: number;
  /** Serving a room off a LAN address with no token. */
  lab_mode?: boolean;
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

/**
 * What the region inside the bounding box is filled as.
 *
 * A face is not a rectangle. The corners of an expanded eyes-to-chin box
 * are shoulder and wall, and masking them tells the run to learn nothing
 * from pixels it should be learning from. Detection produces ellipses;
 * `rect` stays because a hand-drawn box over a sign or a logo genuinely is
 * one, and because every box file written before this existed holds them.
 */
export type BoxShape = "rect" | "ellipse";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  src: BoxSource;
  conf?: number | null;
  /** Absent means `rect` — the daemon omits it for the default. */
  shape?: BoxShape;
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
  /** Which project submitted this. The only identity a lab node has. */
  project: string;
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
  project: string;
  config_path: string;
  progress: { step: number; total: number };
  events: number;
  spec: RunSpec;
  /** How many runs are ahead of this one across the *whole* queue, or -1
   *  once it is no longer waiting. Counted by the daemon, because a client
   *  counting its own filtered list would say "you are next" while four
   *  other people were in front. */
  position?: number;
}

/** One waiting run, as the shared queue lists it. */
export interface QueueEntry {
  id: string;
  project: string;
  name: string;
  model?: string;
  device?: number;
}

export interface JobsResponse {
  jobs: Job[];
  /** Every waiting run on the node, in the order it will start. Not
   *  filtered by project, even when `jobs` is. */
  queue: QueueEntry[];
  running: QueueEntry[];
  depth: number;
  devices: number;
  runner: boolean;
}

// --------------------------------------------------------------------------
// what a finished run produced
// --------------------------------------------------------------------------

/** One `.safetensors` file in a run's output folder. */
export interface Artifact {
  name: string;
  path: string;
  /** The step it was saved at, or null for the final weights. */
  step: number | null;
  final: boolean;
  size: number;
  mtime: number;
}

export interface ArtifactsResponse {
  id: string;
  artifacts: Artifact[];
  /** The `models/loras/<here>` folder these belong in. */
  family: string;
  comfyui: string;
  /** False when there is nothing to publish, or nowhere to publish it. */
  publishable: boolean;
}

export interface PublishResult {
  id: string;
  artifact: string;
  published: string;
  size: number;
}

// --------------------------------------------------------------------------
// projects
// --------------------------------------------------------------------------

/**
 * A named group of dataset folders sharing one training config.
 *
 * The project is also the identity: a lab node has no accounts, so the
 * project a browser has open is what the shared queue lists beside a run.
 */
export interface Project {
  id: string;
  name: string;
  /** Dataset ids, already resolved against the node's registry. */
  datasets: string[];
  dataset_details: Dataset[];
  /** Ids the project holds that the node no longer has registered. Shown
   *  rather than hidden, so a missing folder is a visible fact. */
  missing: string[];
  /** The shared training form. Opaque to the daemon. */
  config: Record<string, unknown>;
  created: number;
  updated: number;
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
  /** Did the backend say which image each step used? Without it there are
   *  no outliers to find, which is not the same as finding none. */
  attributed: boolean;
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

export interface SavedPrompt {
  name: string;
  text: string;
  /** Shipped with the app; cannot be deleted, only saved over. */
  builtin: boolean;
  /** A saved prompt standing over a built-in of the same name. */
  shadows_builtin: boolean;
}

// --------------------------------------------------------------------------
// dataset registration and run planning
// --------------------------------------------------------------------------

export interface FolderEntry {
  path: string;
  name: string;
  images: number;
  has_masks: boolean;
  /** Set when this folder is already registered. */
  dataset_id: string | null;
}

export interface BrowseResponse {
  path: string | null;
  parent: string | null;
  roots: string[];
  entries: FolderEntry[];
}

/** What a run would be, before it is submitted. */
export interface RunPlan {
  dataset: string;
  images: number;
  repeats: number;
  epochs: number;
  steps: number;
  seconds_per_step: number | null;
  seconds: number | null;
  /** "9h 30m", or empty when this node has no history to measure against. */
  duration: string;
  /** Where the rate came from, so an extrapolation is not read as a promise. */
  basis: string;
}

/** Where a run wrote. `*_exists` is checked, not assumed. */
export interface RunFolders {
  id: string;
  output: string;
  output_exists: boolean;
  samples: string;
  samples_exists: boolean;
}

export interface OpenedFolder {
  path: string;
  opened: boolean;
  /** Why not, when it could not be opened - a headless node, usually. */
  detail: string;
}
