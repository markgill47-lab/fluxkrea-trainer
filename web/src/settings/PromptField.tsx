/**
 * The prompt, and the library it can be loaded from and saved into.
 *
 * A caption prompt is arrived at by editing it a dozen times against a real
 * dataset, so the thing that matters is not the textarea — it is that last
 * week's version is still there. Picking a saved prompt fills the box;
 * "Save as" puts the current text back under a name.
 *
 * Built-ins are listed with the saved ones and are not deletable. Saving
 * over a built-in name keeps your version and remembers the original
 * underneath, so a bad edit costs nothing permanent — the row offers
 * "Revert" rather than "Delete" in that case.
 */

import { useState } from "preact/hooks";
import type { SavedPrompt } from "~/api/types";

export function PromptField({
  value,
  prompts,
  onChange,
  onSave,
  onDelete,
  onError,
}: {
  value: string;
  prompts: SavedPrompt[];
  /** Apply prompt text as the active prompt (writes captioner.prompt). */
  onChange(text: string): Promise<string | null>;
  onSave(name: string, text: string): Promise<string | null>;
  onDelete(name: string): Promise<string | null>;
  onError(message: string | null): void;
}) {
  const [draft, setDraft] = useState(value);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [state, setState] = useState<"idle" | "busy" | "saved">("idle");

  // Which saved prompt the current text matches, if any. Compared on text
  // rather than tracked as a selection, so editing a loaded prompt visibly
  // detaches it from its name instead of silently claiming to still be it.
  const current = prompts.find((prompt) => prompt.text.trim() === draft.trim());
  const dirty = draft.trim() !== value.trim();

  async function run(work: () => Promise<string | null>) {
    setState("busy");
    const failure = await work();
    if (failure) {
      onError(failure);
      setState("idle");
      return false;
    }
    setState("saved");
    window.setTimeout(() => setState("idle"), 1600);
    return true;
  }

  async function load(pick: string) {
    const found = prompts.find((prompt) => prompt.name === pick);
    if (!found) return;
    setDraft(found.text);
    await run(() => onChange(found.text));
  }

  async function saveAs() {
    const trimmed = name.trim();
    if (!trimmed) return;
    if (await run(() => onSave(trimmed, draft))) {
      setNaming(false);
      setName("");
    }
  }

  return (
    <div class="field field--prompt">
      <label class="field__label" for="field-prompt">
        Prompt
      </label>

      <div class="field__control">
        <select
          class="field__input"
          value={current?.name ?? ""}
          onChange={(event) => void load((event.target as HTMLSelectElement).value)}
          aria-label="Saved prompts"
        >
          <option value="">
            {dirty ? "— edited —" : current ? current.name : "— custom —"}
          </option>
          {prompts.map((prompt) => (
            <option key={prompt.name} value={prompt.name}>
              {prompt.name}
              {prompt.builtin ? " (built-in)" : ""}
            </option>
          ))}
        </select>

        <span class="field__state" role="status">
          {state === "busy" && <span class="spinner" aria-label="Saving" />}
          {state === "saved" && <span class="field__ok">saved</span>}
        </span>
      </div>

      <textarea
        id="field-prompt"
        class="field__input field__input--area field__prompt-text"
        rows={5}
        value={draft}
        placeholder="(the built-in prompt)"
        onInput={(event) => setDraft((event.target as HTMLTextAreaElement).value)}
        onBlur={() => {
          if (dirty) void run(() => onChange(draft));
        }}
      />

      <div class="field__hint">
        What the model is asked. Empty uses the built-in caption prompt. A prompt
        that lists what to cover should also say <em>write one flowing paragraph</em>,
        or the model may answer with a labelled form.
      </div>

      <div class="prompt-actions">
        {naming ? (
          <>
            <input
              class="field__input"
              placeholder="name, e.g. mara-portrait"
              value={name}
              autofocus
              onInput={(event) => setName((event.target as HTMLInputElement).value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void saveAs();
                if (event.key === "Escape") setNaming(false);
              }}
            />
            <button class="btn btn--accent" onClick={() => void saveAs()} disabled={!name.trim()}>
              Save
            </button>
            <button class="btn btn--ghost" onClick={() => setNaming(false)}>
              Cancel
            </button>
          </>
        ) : (
          <>
            <button class="btn" onClick={() => setNaming(true)} disabled={!draft.trim()}>
              Save as…
            </button>
            {current && !current.builtin && (
              <button
                class="btn btn--ghost"
                onClick={() => void run(() => onDelete(current.name))}
                title={
                  current.shadows_builtin
                    ? "Remove your version; the built-in of this name comes back"
                    : "Delete this saved prompt"
                }
              >
                {current.shadows_builtin ? "Revert" : "Delete"} “{current.name}”
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
