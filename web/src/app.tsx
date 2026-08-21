/**
 * The app shell — doc 09.
 *
 * Top bar with the node selector at far left, because everything the
 * screen shows is scoped to it and so it reads first; connection state at
 * far right, because over an SSH tunnel it is a thing you check. A 48px
 * rail, and a content region that owns its own scrolling. The shell never
 * scrolls.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Dataset, Health, NodeInfo } from "~/api/types";
import { DatasetPicker } from "~/datasets/DatasetPicker";
import { GalleryScreen } from "~/gallery/GalleryScreen";
import { ReviewScreen } from "~/review/ReviewScreen";
import { TrainScreen } from "~/train/TrainScreen";
import { SettingsScreen } from "~/settings/SettingsScreen";

/**
 * The screens a *node-served* client offers.
 *
 * Deliberately no fleet view. The client is served by each node, so a
 * fleet tab would need the node list from somewhere — either the daemon
 * serves it, and every node then knows about every other, or the browser
 * reaches each node directly. Both break doc 06's "client-side
 * aggregation, no coordinator", and since the API is remote code execution
 * scoped to a node, chaining it means one compromised UI reaches all of
 * them.
 *
 * Fleet aggregation stays where the node list belongs: on the operator's
 * own machine, via `fk fleet status`.
 */
type Screen = "datasets" | "review" | "training" | "settings";

/** How often to re-check the daemon. Connection state is UI state. */
const HEALTH_INTERVAL = 10_000;

export function App() {
  const [screen, setScreen] = useState<Screen>("datasets");
  const [health, setHealth] = useState<Health | null>(null);
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [picking, setPicking] = useState(false);

  const poll = useCallback(async () => {
    try {
      setHealth(await api.health());
      setOnline(true);
    } catch (caught) {
      if (!isAbort(caught)) setOnline(false);
    }
  }, []);

  useEffect(() => {
    void poll();
    const timer = setInterval(poll, HEALTH_INTERVAL);
    return () => clearInterval(timer);
  }, [poll]);

  const reloadDatasets = useCallback(async () => {
    try {
      const listing = await api.datasets();
      setDatasets(listing.datasets);
      // Keep the selection if it survived; otherwise fall to the first one.
      setDataset((existing) =>
        existing && listing.datasets.some((entry) => entry.id === existing)
          ? existing
          : (listing.datasets[0]?.id ?? null),
      );
    } catch (caught) {
      if (!isAbort(caught)) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setNode(await api.node());
      } catch (caught) {
        if (!isAbort(caught)) {
          setError(caught instanceof ApiError ? caught.message : String(caught));
        }
      }
    })();
    void reloadDatasets();
  }, [reloadDatasets]);

  const detectors = node
    ? Object.entries(node.detectors)
        .filter(([, ready]) => ready)
        .map(([name]) => name)
    : ["yunet"];

  return (
    <div class="shell">
      <header class="topbar">
        <label class="node-select">
          <span style={{ color: "var(--text-tertiary)" }}>node</span>
          <strong>{node?.name ?? health?.node ?? "…"}</strong>
        </label>
        <span class="topbar__brand">FluxKrea 26</span>

        {datasets.length > 0 && (
          <select
            class="node-select"
            value={dataset ?? ""}
            onChange={(event) => setDataset((event.target as HTMLSelectElement).value)}
            aria-label="Dataset"
          >
            {datasets.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.id}
              </option>
            ))}
          </select>
        )}

        <button
          class="btn btn--ghost"
          onClick={() => setPicking(true)}
          title="Add or remove dataset folders"
        >
          Datasets…
        </button>

        <span class="topbar__spacer" />

        <span
          class={`conn ${online === true ? "conn--up" : online === false ? "conn--down" : ""}`}
          role="status"
        >
          <span class="conn__dot" aria-hidden="true" />
          {online === true ? "connected" : online === false ? "no daemon" : "connecting"}
        </span>
      </header>

      <nav class="rail" aria-label="Sections">
        <RailItem label="Datasets" active={screen === "datasets"} onClick={() => setScreen("datasets")}>
          ▤
        </RailItem>
        <RailItem label="Review" active={screen === "review"} onClick={() => setScreen("review")}>
          ◧
        </RailItem>
        <RailItem label="Training" active={screen === "training"} onClick={() => setScreen("training")}>
          ⏵
        </RailItem>
        <span class="rail__spacer" />
        <RailItem label="Settings" active={screen === "settings"} onClick={() => setScreen("settings")}>
          ⚙
        </RailItem>
      </nav>

      <main class="content">
        {error && <div class="banner">⚠ {error}</div>}

        {screen === "datasets" &&
          (dataset ? (
            <GalleryScreen
              dataset={dataset}
              onError={setError}
              onOpenReview={() => setScreen("review")}
            />
          ) : (
            <NoDataset onAdd={() => setPicking(true)} />
          ))}

        {screen === "review" &&
          (dataset ? (
            <ReviewScreen dataset={dataset} detectors={detectors} onError={setError} />
          ) : (
            <NoDataset onAdd={() => setPicking(true)} />
          ))}

        {screen === "training" && (
          <TrainScreen
            datasets={datasets}
            dataset={dataset}
            onDataset={setDataset}
            onError={setError}
          />
        )}

        {screen === "settings" && <SettingsScreen onError={setError} />}
      </main>

      {picking && (
        <DatasetPicker
          datasets={datasets}
          onClose={() => setPicking(false)}
          onChanged={() => void reloadDatasets()}
        />
      )}
    </div>
  );
}

function NoDataset({ onAdd }: { onAdd(): void }) {
  return (
    <div class="empty">
      <div class="empty__title">No dataset registered</div>
      <div>Point this node at a folder of images to get started.</div>
      <div>
        <button class="btn btn--accent" onClick={onAdd}>
          Add a dataset
        </button>
      </div>
      <div style={{ color: "var(--text-tertiary)" }}>
        or <code class="mono">fk dataset register &lt;path&gt;</code>
      </div>
    </div>
  );
}

function titleOf(screen: Screen): string {
  return {
    datasets: "Dataset gallery",
    review: "Mask review",
    training: "Training monitor",
    settings: "Settings",
  }[screen];
}

function RailItem({
  label,
  active,
  onClick,
  children,
}: {
  label: string;
  active: boolean;
  onClick(): void;
  children: preact.ComponentChildren;
}) {
  return (
    <button
      class="rail__item"
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      {children}
    </button>
  );
}
