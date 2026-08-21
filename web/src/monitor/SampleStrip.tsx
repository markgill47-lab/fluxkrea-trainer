/**
 * The sample strip — doc 08: a horizontal timeline of generated samples
 * grouped by step, click to open full size, lazy-loaded.
 *
 * Doc 10 flags these as the sneaky bandwidth cost: "a run generating a
 * 1024x1024 sample every 400 steps will fill a strip fast." They are
 * therefore laid out small and fetched lazily, with full size only on
 * click.
 */

import { useState } from "preact/hooks";
import type { SampleImage } from "~/api/types";

interface Props {
  samples: SampleImage[];
}

export function SampleStrip({ samples }: Props) {
  const [open, setOpen] = useState<SampleImage | null>(null);

  if (!samples.length) {
    return (
      <section class="samples" aria-label="Samples">
        <div class="empty">
          <div>No samples yet</div>
        </div>
      </section>
    );
  }

  return (
    <section class="samples" aria-label="Samples">
      <div class="samples__scroll">
        {samples.map((sample) => (
          <figure class="sample" key={sample.name} onClick={() => setOpen(sample)}>
            <img src={sample.url} alt="" loading="lazy" decoding="async" />
            <figcaption class="tabular">{sample.step || "—"}</figcaption>
          </figure>
        ))}
      </div>

      {open && (
        <div class="lightbox" onClick={() => setOpen(null)} role="dialog" aria-modal="true">
          <img src={open.url} alt={`sample at step ${open.step}`} />
          <span class="lightbox__caption mono">
            {open.name} · step {open.step}
          </span>
        </div>
      )}
    </section>
  );
}
