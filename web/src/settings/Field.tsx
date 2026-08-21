/**
 * One setting, saved on blur.
 *
 * The row owns its own state machine — editing, saving, saved, failed —
 * because a screen-wide status line cannot say *which* field failed, and
 * that is the only thing worth knowing when one does.
 *
 * A value only saves when it actually changed. Tabbing through a form
 * should not rewrite the config file once per field.
 */

import { useEffect, useRef, useState } from "preact/hooks";

export interface Option {
  value: string;
  label: string;
  disabled?: boolean;
}

type State = "idle" | "saving" | "saved" | "failed";

/** How long the ✓ stays up. Long enough to notice, short enough not to nag. */
const CONFIRM_MS = 1600;

export function Field({
  label,
  hint,
  value,
  options,
  type = "text",
  step,
  multiline = false,
  placeholder,
  onSave,
}: {
  label: string;
  hint?: string;
  value: string | number | boolean;
  options?: Option[];
  type?: "text" | "number";
  step?: string;
  multiline?: boolean;
  placeholder?: string;
  /** Resolves to an error message, or null when the write succeeded. */
  onSave(value: string): Promise<string | null>;
}) {
  const asText = String(value);
  const [draft, setDraft] = useState(asText);
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  // The server is the source of truth: a rejected write leaves the config
  // unchanged, and this pulls the row back to what the node actually holds.
  useEffect(() => setDraft(asText), [asText]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function commit(next: string) {
    if (next === asText) {
      setState("idle");
      setError(null);
      return;
    }
    setState("saving");
    const failure = await onSave(next);
    if (failure) {
      setState("failed");
      setError(failure);
      setDraft(asText); // show what the node still has, not what was refused
      return;
    }
    setState("saved");
    setError(null);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState("idle"), CONFIRM_MS);
  }

  const id = `field-${label.toLowerCase().replace(/\s+/g, "-")}`;
  const busy = state === "saving";

  return (
    <div class={`field${state === "failed" ? " field--failed" : ""}`}>
      <label class="field__label" for={id}>
        {label}
      </label>

      <div class="field__control">
        {options ? (
          <select
            id={id}
            class="field__input"
            value={draft}
            disabled={busy}
            onChange={(event) => {
              const next = (event.target as HTMLSelectElement).value;
              setDraft(next);
              void commit(next);
            }}
          >
            {options.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>
        ) : multiline ? (
          <textarea
            id={id}
            class="field__input field__input--area"
            rows={3}
            value={draft}
            disabled={busy}
            placeholder={placeholder}
            onInput={(event) => setDraft((event.target as HTMLTextAreaElement).value)}
            onBlur={() => void commit(draft)}
          />
        ) : (
          <input
            id={id}
            class="field__input"
            type={type}
            step={step}
            value={draft}
            disabled={busy}
            placeholder={placeholder}
            onInput={(event) => setDraft((event.target as HTMLInputElement).value)}
            onBlur={() => void commit(draft)}
            onKeyDown={(event) => {
              if (event.key === "Enter") (event.target as HTMLInputElement).blur();
              if (event.key === "Escape") {
                setDraft(asText);
                (event.target as HTMLInputElement).blur();
              }
            }}
          />
        )}

        <span class="field__state" role="status">
          {state === "saving" && <span class="spinner" aria-label="Saving" />}
          {state === "saved" && <span class="field__ok">saved</span>}
        </span>
      </div>

      {error ? (
        <div class="field__hint field__hint--error">{error}</div>
      ) : hint ? (
        <div class="field__hint">{hint}</div>
      ) : null}
    </div>
  );
}
