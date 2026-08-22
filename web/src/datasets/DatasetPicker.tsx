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
 *
 * The registered list is scoped to the open project, and adding a folder
 * puts it in that project. "Remove" is therefore two different promises
 * depending on where the node stands: it takes the dataset out of this
 * project, and the shell decides whether that also means deregistering it
 * from the node. Both leave every file alone.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { BrowseResponse, Dataset } from "~/api/types";

/** Is this folder already inside one of the node's dataset roots? */
function withinRoots(path: string, roots: string[]): boolean {
  const target = path.replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
  return roots.some((root) => {
    const base = root.replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
    return target === base || target.startsWith(`${base}/`);
  });
}

export function DatasetPicker({
  datasets,
  onClose,
  onChanged,
  onRemove,
}: {
  datasets: Dataset[];
  onClose(): void;
  /** Called after a register or removal, so the shell can reload its list.
   *  The id of a newly registered dataset is passed so the shell can add
   *  it to the open project without a second browse. */
  onChanged(added?: string): void;
  /** Take a dataset out of the open project. Registration is untouched. */
  onRemove(dataset: string): Promise<void>;
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

  /**
   * Run one picker action and refresh.
   *
   * *added* is passed explicitly rather than sniffed out of the response.
   * Both register and forget answer with a dataset id, so reading the id
   * off whatever came back would have handed the shell a *removed*
   * dataset to add to the project.
   */
  async function act(key: string, work: () => Promise<string | void>) {
    setBusy(key);
    setError(null);
    try {
      const added = await work();
      onChanged(added || undefined);
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
          <h3 class="modal__subtitle">In this project</h3>
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
                    onClick={() =>
                      void act(dataset.id, async () => {
                        await onRemove(dataset.id);
                      })
                    }
                    title={
                      "Take this dataset out of the project. It stays registered on " +
                      "the node, and the folder and its files are left alone."
                    }
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
                      onClick={() =>
                        void act(entry.path, async (): Promise<string> => {
                          // A folder outside the configured roots cannot be
                          // registered until it is inside one. Adding the
                          // root is the step that used to be missing
                          // entirely: the work was on another drive, and
                          // there was no way to say so from in here.
                          if (!withinRoots(entry.path, listing?.roots ?? [])) {
                            await api.addRoot(entry.path);
                          }
                          const registered = await api.registerDataset(entry.path);
                          return registered.id;
                        })
                      }
                      title={
                        entry.images === 0
                          ? "No images in this folder — open it to find the one that has them"
                          : withinRoots(entry.path, listing?.roots ?? [])
                            ? "Register this folder as a dataset"
                            : "Outside this node's dataset roots — adding it will widen them"
                      }
                    >
                      {withinRoots(entry.path, listing?.roots ?? []) ? "Add" : "Add + allow"}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}

          <p class="modal__note">
            Datasets may live anywhere; the roots below are what this node will
            open without being asked again.{" "}
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
