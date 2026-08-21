/**
 * The API client. Every screen goes through this and nothing else.
 *
 * Two things it takes seriously, both because everything here is viewed
 * over an SSH tunnel that will drop (doc 10):
 *
 * - **Requests are cancellable.** Moving to the next image must abort the
 *   previous image's fetch, or a fast review pass queues megabytes it will
 *   never look at.
 * - **Connection state is UI state.** A failed request is reported, not
 *   swallowed; the shell shows it.
 */

import type {
  BoxesResponse,
  Dataset,
  Health,
  ItemsResponse,
  ModelInfo,
  NodeInfo,
  ReviewProgress,
  Task,
  ValidationReport,
} from "./types";

const API = "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Thrown when a request was deliberately abandoned. Not a failure. */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export interface RequestOptions {
  signal?: AbortSignal;
  params?: Record<string, string | number | boolean | undefined>;
}

/** The base URL. Empty in normal use; set when driving another node. */
let base = "";

export function setBase(url: string): void {
  base = url.replace(/\/$/, "");
}

/** A token, when the daemon is bound wider than loopback. */
let token: string | null = null;

export function setToken(value: string | null): void {
  token = value;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const url = new URL(`${base}${API}${path}`, window.location.href);
  for (const [key, value] of Object.entries(options.params ?? {})) {
    if (value !== undefined) url.searchParams.set(key, String(value));
  }

  const headers: Record<string, string> = {};
  if (body !== undefined) headers["content-type"] = "application/json";
  if (token) headers["x-fluxkrea-token"] = token;

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: options.signal,
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(`cannot reach the daemon: ${String(error)}`, 0);
  }

  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string; detail?: unknown };
    if (payload.error) return payload.error;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail) && payload.detail.length) {
      const first = payload.detail[0] as { msg?: string };
      return first.msg ?? `${response.status}`;
    }
  } catch {
    /* fall through to the status */
  }
  return `${response.status} ${response.statusText}`;
}

// --------------------------------------------------------------------------
// endpoints
// --------------------------------------------------------------------------

export const api = {
  health: (o?: RequestOptions) => request<Health>("GET", "/health", undefined, o),
  node: (o?: RequestOptions) => request<NodeInfo>("GET", "/node", undefined, o),
  models: (o?: RequestOptions) =>
    request<{ models: ModelInfo[]; backends: NodeInfo["backends"] }>(
      "GET",
      "/models",
      undefined,
      o,
    ),

  datasets: (o?: RequestOptions) =>
    request<{ datasets: Dataset[] }>("GET", "/datasets", undefined, o),

  registerDataset: (path: string, name?: string, o?: RequestOptions) =>
    request<Dataset>("POST", "/datasets", { path, name }, o),

  items: (dataset: string, o?: RequestOptions) =>
    request<ItemsResponse>("GET", `/datasets/${dataset}/items`, undefined, o),

  review: (dataset: string, o?: RequestOptions) =>
    request<ReviewProgress>("GET", `/datasets/${dataset}/review`, undefined, o),

  validate: (dataset: string, requireMasks = false, o?: RequestOptions) =>
    request<ValidationReport>("GET", `/datasets/${dataset}/validate`, undefined, {
      ...o,
      params: { require_masks: requireMasks },
    }),

  boxes: (dataset: string, stem: string, o?: RequestOptions) =>
    request<BoxesResponse>("GET", `/datasets/${dataset}/items/${stem}/boxes`, undefined, o),

  /**
   * Replace one image's boxes. The remote review pass (doc 06).
   *
   * Box edits are optimistic in the UI and saved on navigation, so this is
   * called once per image rather than once per drag.
   */
  putBoxes: (
    dataset: string,
    stem: string,
    boxes: BoxesResponse["boxes"],
    reviewed: boolean,
    o?: RequestOptions,
  ) =>
    request<BoxesResponse>(
      "PUT",
      `/datasets/${dataset}/items/${stem}/boxes`,
      { boxes, reviewed },
      o,
    ),

  caption: (dataset: string, stem: string, o?: RequestOptions) =>
    request<{ stem: string; caption: string }>(
      "GET",
      `/datasets/${dataset}/items/${stem}/caption`,
      undefined,
      o,
    ),

  putCaption: (dataset: string, stem: string, caption: string, o?: RequestOptions) =>
    request<{ stem: string; caption: string }>(
      "PUT",
      `/datasets/${dataset}/items/${stem}/caption`,
      { caption },
      o,
    ),

  /** Quality is derived metadata; null clears it. */
  putQuality: (dataset: string, stem: string, quality: string | null, o?: RequestOptions) =>
    request<{ stem: string; quality: string | null }>(
      "PUT",
      `/datasets/${dataset}/items/${stem}/quality`,
      { quality },
      o,
    ),

  runOperation: (
    dataset: string,
    operation: string,
    options: Record<string, unknown> = {},
    o?: RequestOptions,
  ) => request<Task>("POST", `/datasets/${dataset}/ops/${operation}`, options, o),

  task: (id: string, o?: RequestOptions) => request<Task>("GET", `/tasks/${id}`, undefined, o),

  cancelTask: (id: string, o?: RequestOptions) =>
    request<{ id: string; cancelling: boolean }>("DELETE", `/tasks/${id}`, undefined, o),
};

// --------------------------------------------------------------------------
// asset URLs
// --------------------------------------------------------------------------

/**
 * Image URLs are built, not fetched — they go straight into `<img>` and
 * `createImageBitmap`, so the browser's own cache and range requests do
 * the work rather than anything here.
 */
export const assets = {
  image: (dataset: string, stem: string) => `${base}${API}/datasets/${dataset}/items/${stem}/image`,

  /**
   * Content-addressed thumbnail. The digest changes when the file does,
   * so the response is immutable and no invalidation logic is needed
   * anywhere (doc 10).
   */
  thumb: (dataset: string, stem: string, size: 160 | 480 = 160, token?: string) =>
    `${base}${API}/datasets/${dataset}/items/${stem}/thumb?size=${size}` +
    (token ? `&v=${token}` : ""),

  mask: (dataset: string, stem: string) => `${base}${API}/datasets/${dataset}/items/${stem}/mask`,

  preview: (dataset: string, stem: string) =>
    `${base}${API}/datasets/${dataset}/items/${stem}/preview`,
};

/** SSE URL for a task's event stream, resumable by index. */
export function taskEvents(id: string, since = -1): string {
  const url = new URL(`${base}${API}/tasks/${id}/events`, window.location.href);
  if (since >= 0) url.searchParams.set("since", String(since));
  return url.toString();
}
