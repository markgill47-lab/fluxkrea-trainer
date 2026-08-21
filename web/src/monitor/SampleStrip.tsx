/**
 * The sample strip — doc 08: a horizontal timeline of generated samples
 * grouped by step, click to open full size, lazy-loaded.
 *
 * Doc 10 flags these as the sneaky bandwidth cost: "a run generating a
 * 1024x1024 sample every 400 steps will fill a strip fast." They are
 * therefore laid out small and fetched lazily, with full size only on
 * click.
 */

import { useCallback, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { SampleImage } from "~/api/types";

interface Props {
  samples: SampleImage[];
  /** The run these belong to, so the folder can be opened. */
  jobId?: string | null;
}

export function SampleStrip({ samples, jobId = null }: Props) {
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
      {jobId && <SamplesFolder jobId={jobId} />}
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


/**
 * Open the samples folder — on the node, which over a tunnel is not the
 * machine you are looking at.
 *
 * So the path is shown as well as offered, and copying it is one click.
 * A headless fleet node cannot open anything and says so rather than
 * failing silently, which is why the endpoint answers 200 with a reason.
 */
function SamplesFolder({ jobId }: { jobId: string }) {
  const [message, setMessage] = useState<string | null>(null);
  const [path, setPath] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const reveal = useCallback(async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.openRunFolder(jobId, "samples");
      setPath(result.path);
      setMessage(result.opened ? result.detail : `${result.detail}`);
    } catch (error) {
      if (!isAbort(error)) {
        setMessage(error instanceof ApiError ? error.message : String(error));
      }
    } finally {
      setBusy(false);
    }
  }, [jobId]);

  const copy = useCallback(async () => {
    if (!path) return;
    try {
      await navigator.clipboard.writeText(path);
      setMessage("path copied");
    } catch {
      // Clipboard access needs a secure context, which a plain-http
      // tunnel is not. The path is on screen either way.
      setMessage("select the path above to copy it");
    }
  }, [path]);

  return (
    <div class="samples__bar">
      <button class="btn btn--ghost" onClick={() => void reveal()} disabled={busy}>
        {busy ? <span class="spinner" /> : null}
        Open samples folder
      </button>
      {path && (
        <button
          class="samples__path mono"
          onClick={() => void copy()}
          title="Copy this path"
        >
          {path}
        </button>
      )}
      {message && <span class="samples__note">{message}</span>}
    </div>
  );
}
