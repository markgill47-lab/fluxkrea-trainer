/**
 * The training configuration — v1's training tab, as a form that submits a
 * `RunSpec`.
 *
 * **Steps are derived, not typed.** `images x repeats x epochs` is how
 * every trainer counts them, and the number that matters is the total: it
 * is the difference between noticing a two-hour run and discovering a
 * twenty-hour one at midnight. The arithmetic and the duration estimate
 * come from `POST /jobs/plan` rather than being recomputed here, so the
 * CLI and this screen cannot disagree about them.
 *
 * **The estimate is measured or absent.** The node derives seconds-per-step
 * from runs that already finished on it and says which; with no history it
 * shows nothing rather than a number someone would plan an evening around.
 *
 * **Everything locks while a run is going.** Not because the form would
 * corrupt anything — a submitted spec is a copy — but because a field that
 * accepts an edit which changes nothing is a lie about what the run is
 * doing.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Dataset, ModelInfo, RunPlan } from "~/api/types";

export interface FormState {
  model: string;
  dataset: string;
  name: string;
  device: number;
  epochs: number;
  repeats: number;
  learningRate: number;
  networkDim: number;
  networkAlpha: number;
  saveEvery: number;
  scheduler: string;
  resolution: number;
  batchSize: number;
  masked: boolean;
  maskMin: number;
  sampleEnabled: boolean;
  samplePrompt: string;
  sampleEvery: number;
  sampleWidth: number;
  sampleHeight: number;
  seed: number;
  randomSeed: boolean;
}

export const DEFAULTS: FormState = {
  model: "",
  dataset: "",
  name: "",
  device: 0,
  epochs: 6,
  repeats: 10,
  learningRate: 0.0002,
  networkDim: 32,
  networkAlpha: 16,
  saveEvery: 800,
  // cosine by default: doc 05's note is that constant can diverge on long
  // runs, and the long runs are the expensive ones to lose.
  scheduler: "cosine",
  resolution: 1024,
  batchSize: 1,
  masked: false,
  maskMin: 0,
  sampleEnabled: true,
  samplePrompt: "",
  sampleEvery: 400,
  sampleWidth: 512,
  sampleHeight: 768,
  seed: 42,
  randomSeed: true,
};

const SCHEDULERS = ["cosine", "constant", "linear", "constant_with_warmup"];

/** Turn the form into the body `POST /jobs` takes. */
export function toSpec(form: FormState, steps: number, maskPath: string): Record<string, unknown> {
  const prompts = form.sampleEnabled
    ? form.samplePrompt
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
    : [];

  return {
    model: form.model,
    dataset: form.dataset,
    name: form.name.trim(),
    device: form.device,
    steps,
    batch_size: form.batchSize,
    learning_rate: form.learningRate,
    network_dim: form.networkDim,
    network_alpha: form.networkAlpha,
    resolution: form.resolution,
    // The one line doc 04 exists to produce. Empty means an unmasked run.
    mask_path: form.masked ? maskPath : "",
    mask_min_value: form.maskMin,
    sample_every: prompts.length ? form.sampleEvery : 0,
    save_every: form.saveEvery,
    // A fixed seed makes samples comparable across checkpoints; a random
    // one shows what the network does rather than what one noise map does.
    seed: form.randomSeed ? null : form.seed,
    extra: {
      lr_scheduler: form.scheduler,
      sample_prompts: prompts,
      sample_width: form.sampleWidth,
      sample_height: form.sampleHeight,
      epochs: form.epochs,
      repeats: form.repeats,
    },
  };
}

export function TrainForm({
  datasets,
  models,
  devices,
  locked,
  onSubmitted,
  onError,
}: {
  datasets: Dataset[];
  models: ModelInfo[];
  devices: number;
  /** A run is going. Every control goes read-only. */
  locked: boolean;
  onSubmitted(jobId: string): void;
  onError(message: string | null): void;
}) {
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [plan, setPlan] = useState<RunPlan | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  // Pick something real the moment the lists arrive, rather than showing a
  // form that cannot be submitted.
  useEffect(() => {
    setForm((current) => ({
      ...current,
      model: current.model || models[0]?.id || "",
      dataset: current.dataset || datasets[0]?.id || "",
    }));
  }, [models, datasets]);

  // The model's own rank is a better starting point than a global default.
  useEffect(() => {
    const chosen = models.find((entry) => entry.id === form.model);
    if (!chosen) return;
    setForm((current) =>
      current.networkDim === DEFAULTS.networkDim && chosen.network_dim
        ? { ...current, networkDim: chosen.network_dim }
        : current,
    );
  }, [form.model, models]);

  const refreshPlan = useCallback(async () => {
    if (!form.dataset) return setPlan(null);
    try {
      setPlan(await api.planRun(form.dataset, form.repeats, form.epochs, form.model));
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    }
  }, [form.dataset, form.repeats, form.epochs, form.model, onError]);

  useEffect(() => {
    void refreshPlan();
  }, [refreshPlan]);

  const steps = plan?.steps ?? 0;
  const chosenDataset = datasets.find((entry) => entry.id === form.dataset);

  async function submit() {
    if (!form.model || !form.dataset || steps < 1) return;
    setSubmitting(true);
    onError(null);
    try {
      const maskPath = chosenDataset ? `${chosenDataset.path}/masks` : "";
      const job = await api.submitJob(toSpec(form, steps, maskPath));
      if (job.warning) onError(job.warning);
      onSubmitted(job.id);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  const disabled = locked || submitting;

  return (
    <form class={`train${locked ? " train--locked" : ""}`} onSubmit={(e) => e.preventDefault()}>
      {locked && (
        <div class="train__lock" role="status">
          A run is going on this node. Settings are locked until it finishes or is cancelled.
        </div>
      )}

      {/* -- duration --------------------------------------------------- */}
      <section class="panel">
        <h2 class="panel__title">Training duration</h2>
        <p class="panel__note">Steps are calculated from the dataset, not typed.</p>

        <div class="train__sum">
          <span class="train__sum-terms tabular">
            {plan ? `${plan.images} images × ${plan.repeats} repeats × ${plan.epochs} epochs` : "…"}
          </span>
          <span class="train__sum-total tabular">{steps.toLocaleString()}</span>
          <span class="train__sum-unit">steps</span>
          {plan?.duration ? (
            <span class="train__sum-time" title={plan.basis}>
              ≈ {plan.duration}
            </span>
          ) : (
            <span class="train__sum-time train__sum-time--unknown" title={plan?.basis ?? ""}>
              duration unknown
            </span>
          )}
          <button
            type="button"
            class="btn btn--ghost"
            onClick={() => void refreshPlan()}
            disabled={disabled}
          >
            Refresh
          </button>
        </div>
        {plan && <div class="train__basis">{plan.basis}</div>}

        <Row label="Dataset">
          <select
            class="field__input"
            value={form.dataset}
            disabled={disabled}
            onChange={(e) => set("dataset", (e.target as HTMLSelectElement).value)}
          >
            {datasets.length === 0 && <option value="">no datasets registered</option>}
            {datasets.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.id}
              </option>
            ))}
          </select>
        </Row>
        <Row label="Epochs" hint="full passes through the dataset">
          <Num value={form.epochs} min={1} disabled={disabled} onInput={(v) => set("epochs", v)} />
        </Row>
        <Row label="Samples per image" hint="repeats per image, per epoch">
          <Num value={form.repeats} min={1} disabled={disabled} onInput={(v) => set("repeats", v)} />
        </Row>
      </section>

      {/* -- parameters ------------------------------------------------- */}
      <section class="panel">
        <h2 class="panel__title">Training parameters</h2>

        <Row label="Model">
          <select
            class="field__input"
            value={form.model}
            disabled={disabled}
            onChange={(e) => set("model", (e.target as HTMLSelectElement).value)}
          >
            {models.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label}
              </option>
            ))}
          </select>
        </Row>
        <Row label="Learning rate">
          <Num
            value={form.learningRate}
            step={0.00005}
            disabled={disabled}
            onInput={(v) => set("learningRate", v)}
          />
        </Row>
        <Row label="LoRA rank (dim)" hint="8–16 fast, 32 balanced, 64+ detailed">
          <Num value={form.networkDim} min={1} disabled={disabled} onInput={(v) => set("networkDim", v)} />
        </Row>
        <Row label="LoRA alpha" hint="defaults to the rank when left at it">
          <Num
            value={form.networkAlpha}
            min={1}
            disabled={disabled}
            onInput={(v) => set("networkAlpha", v)}
          />
        </Row>
        <Row label="Resolution">
          <Num value={form.resolution} min={256} step={64} disabled={disabled} onInput={(v) => set("resolution", v)} />
        </Row>
        <Row label="Batch size">
          <Num value={form.batchSize} min={1} disabled={disabled} onInput={(v) => set("batchSize", v)} />
        </Row>
        <Row label="Save every N steps" hint={countHint(steps, form.saveEvery, "checkpoint")}>
          <Num value={form.saveEvery} min={1} disabled={disabled} onInput={(v) => set("saveEvery", v)} />
        </Row>
        <Row label="LR scheduler" hint="cosine is safe; constant can diverge on long runs">
          <select
            class="field__input"
            value={form.scheduler}
            disabled={disabled}
            onChange={(e) => set("scheduler", (e.target as HTMLSelectElement).value)}
          >
            {SCHEDULERS.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </Row>
        <Row label="GPU" hint={devices > 1 ? `${devices} queue slots on this node` : ""}>
          <select
            class="field__input"
            value={String(form.device)}
            disabled={disabled || devices <= 1}
            onChange={(e) => set("device", Number((e.target as HTMLSelectElement).value))}
          >
            {Array.from({ length: Math.max(1, devices) }, (_, index) => (
              <option key={index} value={String(index)}>
                cuda:{index}
              </option>
            ))}
          </select>
        </Row>
      </section>

      {/* -- masking ---------------------------------------------------- */}
      <section class="panel">
        <h2 class="panel__title">Loss masking</h2>
        <p class="panel__note">
          Points the run at the dataset's <code class="mono">masks/</code> folder. An image with
          no matching mask trains unmasked and ai-toolkit says nothing, so this is validated
          before the run starts.
        </p>
        <Row
          label="Masked run"
          hint={chosenDataset ? `${chosenDataset.path}/masks` : "select a dataset first"}
        >
          <Check
            checked={form.masked}
            disabled={disabled || !chosenDataset}
            label="train with the dataset's masks"
            onChange={(v) => set("masked", v)}
          />
        </Row>
        {form.masked && (
          <Row label="Mask min value" hint="0 ignores the masked region entirely">
            <Num value={form.maskMin} step={0.05} disabled={disabled} onInput={(v) => set("maskMin", v)} />
          </Row>
        )}
      </section>

      {/* -- output and samples ----------------------------------------- */}
      <section class="panel">
        <h2 class="panel__title">Output and samples</h2>

        <Row label="Run name" hint="names the output folder; blank generates one">
          <input
            class="field__input"
            value={form.name}
            disabled={disabled}
            placeholder="e.g. mara-v3"
            onInput={(e) => set("name", (e.target as HTMLInputElement).value)}
          />
        </Row>

        <Row
          label="Sample during training"
          hint={
            form.sampleEnabled ? countHint(steps, form.sampleEvery, "sample") : "no samples"
          }
        >
          <Check
            checked={form.sampleEnabled}
            disabled={disabled}
            label="render sample images as it trains"
            onChange={(v) => set("sampleEnabled", v)}
          />
        </Row>

        {form.sampleEnabled && (
          <>
            <Row label="Sample prompts" hint="one per line">
              <textarea
                class="field__input field__input--area"
                rows={3}
                value={form.samplePrompt}
                disabled={disabled}
                placeholder="Gigerstyle: a beautiful alien queen sitting on a throne"
                onInput={(e) => set("samplePrompt", (e.target as HTMLTextAreaElement).value)}
              />
            </Row>
            <Row label="Every N steps">
              <Num value={form.sampleEvery} min={1} disabled={disabled} onInput={(v) => set("sampleEvery", v)} />
            </Row>
            <Row label="Sample size">
              <span class="train__pair">
                <Num value={form.sampleWidth} min={64} step={64} disabled={disabled} onInput={(v) => set("sampleWidth", v)} />
                <span class="train__times">×</span>
                <Num value={form.sampleHeight} min={64} step={64} disabled={disabled} onInput={(v) => set("sampleHeight", v)} />
              </span>
            </Row>
            <Row label="Seed" hint="a fixed seed makes samples comparable across checkpoints">
              <span class="train__pair">
                <Num
                  value={form.seed}
                  disabled={disabled || form.randomSeed}
                  onInput={(v) => set("seed", v)}
                />
                <Check
                  checked={form.randomSeed}
                  disabled={disabled}
                  label="random each time"
                  onChange={(v) => set("randomSeed", v)}
                />
              </span>
            </Row>
          </>
        )}
      </section>

      <div class="train__actions">
        <button
          type="button"
          class="btn btn--accent"
          disabled={disabled || !form.model || !form.dataset || steps < 1}
          onClick={() => void submit()}
        >
          {submitting ? <span class="spinner" /> : null}
          Start training
        </button>
        <span class="train__actions-note">
          {steps > 0
            ? `${steps.toLocaleString()} steps${plan?.duration ? `, about ${plan.duration}` : ""}`
            : "nothing to train — check the dataset"}
        </span>
      </div>
    </form>
  );
}

// --------------------------------------------------------------------------
// small controls
// --------------------------------------------------------------------------

/**
 * "4 checkpoints" — or, when the interval is longer than the run, why there
 * will not be any. A bare "0 checkpoints" states the outcome without the
 * reason, which is the half a person needs to fix it.
 */
function countHint(steps: number, every: number, noun: string): string {
  if (!steps || every < 1) return "";
  const count = Math.floor(steps / every);
  if (count >= 1) return `${count} ${noun}${count === 1 ? "" : "s"} over the run`;
  return `none — ${every.toLocaleString()} is longer than this ${steps.toLocaleString()}-step run`;
}


function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: preact.ComponentChildren;
}) {
  return (
    <div class="field">
      <label class="field__label">{label}</label>
      <div class="field__control">{children}</div>
      {hint ? <div class="field__hint">{hint}</div> : null}
    </div>
  );
}

function Num({
  value,
  min,
  step,
  disabled,
  onInput,
}: {
  value: number;
  min?: number;
  step?: number;
  disabled?: boolean;
  onInput(value: number): void;
}) {
  return (
    <input
      class="field__input"
      type="number"
      value={value}
      min={min}
      step={step}
      disabled={disabled}
      onInput={(event) => {
        const parsed = Number((event.target as HTMLInputElement).value);
        // A half-typed number is NaN for a keystroke; keeping the old value
        // beats resetting the field to zero under the cursor.
        if (!Number.isNaN(parsed)) onInput(parsed);
      }}
    />
  );
}

function Check({
  checked,
  label,
  disabled,
  onChange,
}: {
  checked: boolean;
  label: string;
  disabled?: boolean;
  onChange(value: boolean): void;
}) {
  return (
    <label class="train__check">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange((event.target as HTMLInputElement).checked)}
      />
      <span>{label}</span>
    </label>
  );
}
