/**
 * Add and remove dataset folders.
 *
 * The client is a browser, so there is no native file dialog and there
 * cannot be one — paths come from `GET /fs/browse`, which is scoped to the
 * configured roots. That scoping is the point rather than a limitation:
 * an unscoped picker would make the API a file browser for the whole
 * machine, and the daemon already runs things.
 *
 * Folders show their image count and whether they are already registered,
 * because "which of these forty folders is the dataset" is the actual
 * question being asked, and a name alone does not answer it.
 *
 * Removing forgets the registration. It never touches the folder — the one
 * thing a destructive-sounding button in a browser must not do.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { BrowseResponse, Dataset } from "~/api/types";

export function DatasetPicker({
  datasets,
  onClose,
  onChanged,
}: {
  datasets: Dataset[];
  onClose(): void;
  /** Called after a register or forget, so the shell can reload its list. */
  onChanged(): void;
}) {
  const [listing, setListing] = useState<BrowseResponse | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const browse = useCallback(async (path?: string) => {
    setError(null);
    try {
      setListing(await api.browse(path));
    } catch (caught) {
      if (!isAbort(caught)) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    }
  }, []);

  useEffect(() => {
    void browse();
  }, [browse]);

  // Escape closes, because a modal that traps you is worse than no modal.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function act(key: string, work: () => Promise<unknown>) {
    setBusy(key);
    setError(null);
    try {
      await work();
      onChanged();
      await browse(listing?.path ?? undefined);
    } catch (caught) {
      if (!isAbort(caught)) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    } finally {
      setBusy(null);
    }
  }

  return (
    <div class="modal" role="dialog" aria-modal="true" aria-label="Datasets">
      <div class="modal__backdrop" onClick={onClose} />
      <div class="modal__panel">
        <header class="modal__head">
          <span class="modal__title">Datasets</span>
          <span class="topbar__spacer" />
          <button class="btn btn--ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        {error && <div class="banner">⚠ {error}</div>}

        {/* -- registered ------------------------------------------------ */}
        <section class="modal__section">
          <h3 class="modal__subtitle">Registered on this node</h3>
          {datasets.length === 0 ? (
            <p class="modal__note">None yet. Add one from the folders below.</p>
          ) : (
            <ul class="picker">
              {datasets.map((dataset) => (
                <li key={dataset.id} class="picker__row">
                  <span class="picker__name">{dataset.id}</span>
                  <span class="picker__path mono" title={dataset.path}>
                    {dataset.path}
                  </span>
                  {!dataset.exists && <span class="chip chip--warn">folder missing</span>}
                  <button
                    class="btn btn--ghost"
                    disabled={busy === dataset.id}
                    onClick={() => void act(dataset.id, () => api.forgetDataset(dataset.id))}
                    title="Forget this dataset. The folder and its files are left alone."
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* -- browse ---------------------------------------------------- */}
        <section class="modal__section modal__section--grow">
          <h3 class="modal__subtitle">Add a folder</h3>

          <div class="picker__crumbs">
            <button class="btn btn--ghost" onClick={() => void browse()} disabled={!listing?.path}>
              roots
            </button>
            {listing?.parent && (
              <button class="btn btn--ghost" onClick={() => void browse(listing.parent ?? undefined)}>
                ↑ up
              </button>
            )}
            <span class="picker__here mono">{listing?.path ?? "configured roots"}</span>
          </div>

          {listing === null ? (
            <div class="empty">
              <span class="spinner" aria-label="Loading" />
            </div>
          ) : listing.entries.length === 0 ? (
            <p class="modal__note">No folders here.</p>
          ) : (
            <ul class="picker picker--scroll">
              {listing.entries.map((entry) => (
                <li key={entry.path} class="picker__row">
                  <button
                    class="picker__open"
                    onClick={() => void browse(entry.path)}
                    title={entry.path}
                  >
                    {entry.name}
                  </button>
                  <span class="picker__meta tabular">
                    {entry.images > 0
                      ? `${entry.images} image${entry.images === 1 ? "" : "s"}`
                      : "—"}
                    {entry.has_masks ? " · masks" : ""}
                  </span>
                  {entry.dataset_id ? (
                    <span class="chip">registered as {entry.dataset_id}</span>
                  ) : (
                    <button
                      class="btn"
                      disabled={busy === entry.path || entry.images === 0}
                      onClick={() => void act(entry.path, () => api.registerDataset(entry.path))}
                      title={
                        entry.images === 0
                          ? "No images in this folder — open it to find the one that has them"
                          : "Register this folder as a dataset"
                      }
                    >
                      Add
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}

          <p class="modal__note">
            Only folders under this node's configured roots are listed.{" "}
            {listing?.roots.length ? (
              <span class="mono">{listing.roots.join(", ")}</span>
            ) : (
              <span>No roots configured, so this starts at your home directory.</span>
            )}
          </p>
        </section>
      </div>
    </div>
  );
}
