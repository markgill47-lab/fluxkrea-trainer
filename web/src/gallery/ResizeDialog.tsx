/**
 * Resize a dataset to a target longest edge, in place.
 *
 * In place means irreversible, so this says what it is about to do before
 * it does it. The breakdown is not a guess: items already carry their
 * dimensions, so the counts of what will shrink, grow and be left alone
 * are computed from the same numbers the operation will read.
 *
 * Three things worth knowing, all of which the dialog says out loud:
 *
 * - **Masks move with their images**, resampled with NEAREST. A mask
 *   resized with anything smoother comes back with grey edges, and a grey
 *   pixel in a mask is a partial loss weight.
 * - **Images smaller than the target are enlarged**, because a bucket of
 *   mixed resolutions trains worse than a few upscaled images. The toggle
 *   turns that off and leaves them alone.
 * - **An image already at the target is copied, not re-encoded**, so
 *   running this twice is not two lossy round trips.
 */

import { useEffect, useMemo, useState } from "preact/hooks";
import type { Item } from "~/api/types";

/** The sizes a FLUX/Krea dataset is actually trained at.
 *
 * No 512. It has never been used here, and an option nobody picks is one
 * more thing to read past on a dialog that exists to make one decision.
 * `fk dataset resize --size N` still takes any number, so the capability
 * is not gone - only the preset.
 */
export const SIZES = [1024, 2048] as const;

export function ResizeDialog({
  items,
  running,
  onRun,
  onClose,
}: {
  items: Item[];
  running: boolean;
  onRun(size: number, upscale: boolean): void;
  onClose(): void;
}) {
  const [size, setSize] = useState<number>(1024);
  const [upscale, setUpscale] = useState(true);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const counts = useMemo(() => {
    let shrink = 0;
    let grow = 0;
    let same = 0;
    let unknown = 0;
    for (const item of items) {
      if (!item.width || !item.height) {
        unknown += 1;
        continue;
      }
      const longest = Math.max(item.width, item.height);
      if (longest > size) shrink += 1;
      else if (longest < size) grow += 1;
      else same += 1;
    }
    return { shrink, grow, same, unknown };
  }, [items, size]);

  const masks = items.filter((item) => item.has_mask).length;
  const changing = upscale
    ? counts.shrink + counts.grow + counts.unknown
    : counts.shrink + counts.unknown;

  return (
    <div class="modal" role="dialog" aria-modal="true" aria-label="Resize dataset">
      <div class="modal__backdrop" onClick={onClose} />
      <div class="modal__panel modal__panel--narrow">
        <header class="modal__head">
          <span class="modal__title">Resize</span>
          <span class="topbar__spacer" />
          <button class="btn btn--ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <section class="modal__section">
          <p class="modal__note modal__note--lead">
            Fits every image's <strong>longest edge</strong> to the target, aspect ratio
            preserved. This rewrites the files in place.
          </p>

          <div class="sizes" role="radiogroup" aria-label="Target longest edge">
            {SIZES.map((option) => (
              <button
                key={option}
                type="button"
                role="radio"
                aria-checked={size === option}
                class="sizes__item"
                disabled={running}
                onClick={() => setSize(option)}
              >
                {option}
                <span class="sizes__unit">px</span>
              </button>
            ))}
          </div>

          <label class="train__check sizes__upscale">
            <input
              type="checkbox"
              checked={upscale}
              disabled={running}
              onChange={(event) => setUpscale((event.target as HTMLInputElement).checked)}
            />
            <span>
              Enlarge images smaller than {size}px
              <span class="sizes__aside">
                {" "}
                — off leaves them alone and reports them
              </span>
            </span>
          </label>
        </section>

        {/* -- what will happen ------------------------------------------ */}
        <section class="modal__section">
          <h3 class="modal__subtitle">What this will do</h3>
          <ul class="tally">
            <Tally n={counts.shrink} label={`shrink to ${size}px`} />
            {upscale ? (
              <Tally n={counts.grow} label={`enlarge to ${size}px`} />
            ) : (
              <Tally n={counts.grow} label="left alone (smaller than the target)" muted />
            )}
            <Tally n={counts.same} label="already correct" muted />
            {counts.unknown > 0 && (
              <Tally n={counts.unknown} label="size not cached yet — will be read from the file" muted />
            )}
            {masks > 0 && (
              <Tally n={masks} label="masks resized with them, NEAREST" muted />
            )}
          </ul>
        </section>

        <footer class="modal__foot">
          <button class="btn btn--ghost" onClick={onClose} disabled={running}>
            Cancel
          </button>
          <span class="topbar__spacer" />
          <span class="modal__note">
            {changing > 0
              ? `${changing} of ${items.length} files will be rewritten`
              : "nothing to do at this size"}
          </span>
          <button
            class="btn btn--accent"
            disabled={running || changing === 0}
            onClick={() => onRun(size, upscale)}
          >
            {running ? <span class="spinner" /> : null}
            Resize to {size}px
          </button>
        </footer>
      </div>
    </div>
  );
}

function Tally({ n, label, muted }: { n: number; label: string; muted?: boolean }) {
  if (n === 0) return null;
  return (
    <li class={`tally__row${muted ? " tally__row--muted" : ""}`}>
      <span class="tally__n tabular">{n}</span>
      <span>{label}</span>
    </li>
  );
}
