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
import { GalleryScreen } from "~/gallery/GalleryScreen";
import { ReviewScreen } from "~/review/ReviewScreen";

type Screen = "datasets" | "review" | "training" | "fleet";

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

  useEffect(() => {
    (async () => {
      try {
        const [info, listing] = await Promise.all([api.node(), api.datasets()]);
        setNode(info);
        setDatasets(listing.datasets);
        setDataset((existing) => existing ?? listing.datasets[0]?.id ?? null);
      } catch (caught) {
        if (!isAbort(caught)) {
          setError(caught instanceof ApiError ? caught.message : String(caught));
        }
      }
    })();
  }, []);

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
        <RailItem label="Fleet" active={screen === "fleet"} onClick={() => setScreen("fleet")}>
          ⛓
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
            <NoDataset />
          ))}

        {screen === "review" &&
          (dataset ? (
            <ReviewScreen dataset={dataset} detectors={detectors} onError={setError} />
          ) : (
            <NoDataset />
          ))}

        {(screen === "training" || screen === "fleet") && (
          <div class="empty">
            <div class="empty__title">{titleOf(screen)}</div>
            <div>Not built yet.</div>
          </div>
        )}
      </main>
    </div>
  );
}

function NoDataset() {
  return (
    <div class="empty">
      <div class="empty__title">No dataset registered</div>
      <div>
        Register one with <code class="mono">fk dataset register &lt;path&gt;</code>
      </div>
    </div>
  );
}

function titleOf(screen: Screen): string {
  return { datasets: "Dataset gallery", review: "Mask review", training: "Training monitor", fleet: "Fleet" }[
    screen
  ];
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
