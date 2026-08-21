/**
 * Fixtures and a fake API.
 *
 * Tests drive components through the real `api` object with its methods
 * replaced, rather than through a mocked `fetch`. The client's job is to
 * call the right endpoint with the right arguments; asserting on that
 * directly says more than asserting on a URL string, and it does not
 * quietly pass when a method is renamed.
 */

import { vi } from "vitest";
import { api } from "~/api/client";
import type { Dataset, Item, Job, ModelInfo, RunPlan } from "~/api/types";

export function dataset(id: string, path = `D:/data/${id}`): Dataset {
  return { id, path, name: id, exists: true };
}

export function item(stem: string, width = 1024, height = 768, extra: Partial<Item> = {}): Item {
  return {
    stem,
    filename: `${stem}.png`,
    caption: "a caption",
    has_caption: true,
    has_mask: false,
    quality: null,
    boxes: 0,
    reviewed: false,
    width,
    height,
    token: `${stem}-token`,
    ...extra,
  };
}

export function model(id: string, label = id): ModelInfo {
  return {
    id,
    arch: id,
    label,
    network_dim: 32,
    network_alpha: 32,
    low_vram: false,
    guidance_scale: 1,
    text_encoder: "",
    notes: "",
  };
}

export function job(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    status: "running",
    created: 0,
    started: 1,
    finished: null,
    error: "",
    device: 0,
    config_path: "",
    progress: { step: 10, total: 100 },
    events: 0,
    spec: {
      model: "flux2",
      dataset: "api",
      name: "a-run",
      output: "",
      device: 0,
      steps: 100,
      batch_size: 1,
      learning_rate: 0.0002,
      network_dim: 32,
      network_alpha: 32,
      resolution: 1024,
      mask_path: "",
      mask_min_value: 0,
      sample_every: 0,
      save_every: 0,
      seed: null,
      extra: {},
    },
    ...overrides,
  };
}

export function plan(images: number, repeats = 10, epochs = 6): RunPlan {
  const steps = images * repeats * epochs;
  return {
    dataset: "",
    images,
    repeats,
    epochs,
    steps,
    seconds_per_step: 0.5,
    seconds: steps * 0.5,
    duration: "1h",
    basis: "measured from the last 1 run on this node",
  };
}

/**
 * Replace the API surface a test needs. Everything not named is left
 * throwing, so a component reaching for an endpoint the test did not
 * anticipate fails loudly instead of hanging on an unresolved promise.
 */
export function fakeApi(overrides: Partial<typeof api>): void {
  for (const key of Object.keys(api) as (keyof typeof api)[]) {
    const replacement = overrides[key];
    vi.spyOn(api, key).mockImplementation(
      (replacement ??
        (() => {
          throw new Error(`api.${String(key)} was called but not stubbed by this test`);
        })) as never,
    );
  }
}

/** Resolves once every pending microtask and timer-free promise has run. */
export const settled = () => new Promise((resolve) => setTimeout(resolve, 0));
