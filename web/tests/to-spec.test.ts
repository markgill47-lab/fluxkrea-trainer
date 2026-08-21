/**
 * `toSpec` — the form's only output, and the thing a wrong run is made of.
 *
 * Pure, so these are cheap; and worth having because the mistakes it can
 * make are silent. A run with `mask_path` set when masking was off trains
 * against masks nobody asked for; a run with it empty when masking was on
 * trains unmasked and ai-toolkit says nothing.
 */

import { describe, expect, it } from "vitest";
import { DEFAULTS, type FormState, toSpec } from "~/train/TrainForm";

const form = (overrides: Partial<FormState> = {}): FormState => ({
  ...DEFAULTS,
  model: "krea2",
  dataset: "poses",
  ...overrides,
});

describe("masking", () => {
  it("sends the mask path only when the run is masked", () => {
    expect(toSpec(form({ masked: true }), 100, "D:/data/poses/masks").mask_path).toBe(
      "D:/data/poses/masks",
    );
  });

  it("sends an empty mask path when it is not", () => {
    // Not "omits" - empty is what the backend reads as unmasked, and an
    // absent key would fall through to the dataclass default anyway.
    expect(toSpec(form({ masked: false }), 100, "D:/data/poses/masks").mask_path).toBe("");
  });
});

describe("samples", () => {
  it("turns sampling off when there are no prompts, whatever the checkbox says", () => {
    const spec = toSpec(form({ sampleEnabled: true, samplePrompt: "   " }), 100, "");
    expect(spec.sample_every).toBe(0);
    expect((spec.extra as Record<string, unknown>).sample_prompts).toEqual([]);
  });

  it("takes one prompt per line and drops the blanks", () => {
    const spec = toSpec(
      form({ sampleEnabled: true, samplePrompt: "a queen\n\n  a knight  \n" }),
      100,
      "",
    );
    expect((spec.extra as Record<string, unknown>).sample_prompts).toEqual(["a queen", "a knight"]);
    expect(spec.sample_every).toBe(DEFAULTS.sampleEvery);
  });

  it("discards prompts entirely when sampling is switched off", () => {
    const spec = toSpec(form({ sampleEnabled: false, samplePrompt: "a queen" }), 100, "");
    expect((spec.extra as Record<string, unknown>).sample_prompts).toEqual([]);
    expect(spec.sample_every).toBe(0);
  });
});

describe("the seed", () => {
  it("sends null for a random seed", () => {
    // null, not 0 - zero is a legitimate fixed seed.
    expect(toSpec(form({ randomSeed: true, seed: 42 }), 100, "").seed).toBeNull();
  });

  it("sends the number for a fixed one, including zero", () => {
    expect(toSpec(form({ randomSeed: false, seed: 42 }), 100, "").seed).toBe(42);
    expect(toSpec(form({ randomSeed: false, seed: 0 }), 100, "").seed).toBe(0);
  });
});

describe("what rides in extra", () => {
  it("carries the settings the backend reads from extra", () => {
    const extra = toSpec(form({ scheduler: "constant" }), 100, "").extra as Record<string, unknown>;
    expect(extra.lr_scheduler).toBe("constant");
    expect(extra.sample_width).toBe(DEFAULTS.sampleWidth);
    expect(extra.sample_height).toBe(DEFAULTS.sampleHeight);
  });

  it("records the epochs and repeats the step count came from", () => {
    // The steps are what the trainer runs; these are how a person reads
    // that number back six weeks later.
    const extra = toSpec(form({ epochs: 6, repeats: 10 }), 3420, "").extra as Record<
      string,
      unknown
    >;
    expect(extra.epochs).toBe(6);
    expect(extra.repeats).toBe(10);
  });
});

describe("the rest", () => {
  it("passes the derived step count through rather than recomputing it", () => {
    expect(toSpec(form(), 2820, "").steps).toBe(2820);
  });

  it("trims the run name", () => {
    expect(toSpec(form({ name: "  mara-v3  " }), 100, "").name).toBe("mara-v3");
  });

  it("carries the model and dataset verbatim", () => {
    const spec = toSpec(form({ model: "flux2-klein-9b", dataset: "blizzard-training" }), 100, "");
    expect(spec.model).toBe("flux2-klein-9b");
    expect(spec.dataset).toBe("blizzard-training");
  });
});
