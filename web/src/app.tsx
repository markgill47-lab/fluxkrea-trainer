/**
 * The app shell — doc 09.
 *
 * Top bar with the node selector at far left, because everything the
 * screen shows is scoped to it and so it reads first; connection state at
 * far right, because over an SSH tunnel it is a thing you check. A 48px
 * rail, and a content region that owns its own scrolling. The shell never
 * scrolls.
 *
 * Between the node and the dataset sits the **project**, because the
 * dataset list is scoped to it. On a lab node several students share one
 * daemon, so the shell shows one project's datasets rather than every
 * folder anybody has registered — and the project is also the identity a
 * submitted run carries into the shared queue.
 */

import { useCallback, useEffect, useMemo, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Dataset, Health, NodeInfo, Project } from "~/api/types";
import { DatasetPicker } from "~/datasets/DatasetPicker";
import { GalleryScreen } from "~/gallery/GalleryScreen";
import { ProjectGate, rememberProject, storedProject } from "~/projects/ProjectGate";
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
type Screen = "datasets" | "masks" | "training" | "settings";

/** How often to re-check the daemon. Connection state is UI state. */
const HEALTH_INTERVAL = 10_000;

export function App() {
  const [screen, setScreen] = useState<Screen>("datasets");
  const [health, setHealth] = useState<Health | null>(null);
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [projectId, setProjectId] = useState<string | null>(storedProject);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gateNotice, setGateNotice] = useState<string | null>(null);
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

  const fail = useCallback((caught: unknown) => {
    if (!isAbort(caught)) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    }
  }, []);

  // -- projects ------------------------------------------------------------

  const reloadProjects = useCallback(async () => {
    try {
      const payload = await api.projects();
      setProjects(payload.projects);
      return payload.projects;
    } catch (caught) {
      fail(caught);
      // An empty list rather than null: null means "not asked yet", and
      // leaving it there would spin forever on a daemon that cannot answer.
      setProjects([]);
      return [];
    }
  }, [fail]);

  useEffect(() => {
    void reloadProjects();
  }, [reloadProjects]);

  const project = useMemo(
    () => (projects ?? []).find((entry) => entry.id === projectId) ?? null,
    [projects, projectId],
  );

  // A project deleted from another browser leaves this one holding an id
  // that no longer resolves. Say so on the way back to the gate rather
  // than silently reopening somebody else's project.
  useEffect(() => {
    if (!projects || !projectId || project) return;
    setGateNotice(`The project you had open (${projectId}) is no longer on this node.`);
    setProjectId(null);
    rememberProject(null);
  }, [projects, projectId, project]);

  const openProject = useCallback((id: string) => {
    setGateNotice(null);
    setProjectId(id);
    rememberProject(id);
    setScreen("datasets");
    setDataset(null);
  }, []);

  const closeProject = useCallback(() => {
    setProjectId(null);
    rememberProject(null);
    setDataset(null);
    setGateNotice(null);
    void reloadProjects();
  }, [reloadProjects]);

  // -- datasets ------------------------------------------------------------

  const reloadDatasets = useCallback(async () => {
    const [listing, current] = await Promise.all([
      api.datasets().catch((caught) => {
        fail(caught);
        return null;
      }),
      projectId
        ? api.project(projectId).catch(() => null)
        : Promise.resolve(null),
    ]);
    if (!listing) return;

    // Scoped to the project. `current.datasets` is already resolved
    // against the registry by the daemon, so a folder somebody deregistered
    // does not appear here as a row that 404s on click.
    const allowed = current ? new Set(current.datasets) : null;
    const visible = allowed
      ? listing.datasets.filter((entry) => allowed.has(entry.id))
      : listing.datasets;

    setDatasets(visible);
    setProjects((existing) =>
      existing && current
        ? existing.map((entry) => (entry.id === current.id ? current : entry))
        : existing,
    );
    // Keep the selection if it survived; otherwise fall to the first one.
    setDataset((existing) =>
      existing && visible.some((entry) => entry.id === existing)
        ? existing
        : (visible[0]?.id ?? null),
    );
  }, [projectId, fail]);

  useEffect(() => {
    (async () => {
      try {
        setNode(await api.node());
      } catch (caught) {
        fail(caught);
      }
    })();
  }, [fail]);

  useEffect(() => {
    void reloadDatasets();
  }, [reloadDatasets]);

  /** Register a folder and put it in the open project, in one step. */
  const onDatasetsChanged = useCallback(
    async (added?: string) => {
      if (added && projectId) {
        try {
          await api.addProjectDataset(projectId, added);
        } catch (caught) {
          fail(caught);
        }
      }
      await reloadDatasets();
    },
    [projectId, reloadDatasets, fail],
  );

  const detectors = node
    ? Object.entries(node.detectors)
        .filter(([, ready]) => ready)
        .map(([name]) => name)
    : ["yunet"];

  // Nothing renders until the project list has arrived: guessing wrong for
  // one frame means opening somebody else's dataset.
  if (projects === null) {
    return (
      <div class="empty">
        <span class="spinner" aria-label="Loading" />
      </div>
    );
  }

  if (!project) {
    return (
      <ProjectGate
        projects={projects}
        notice={gateNotice}
        onOpen={openProject}
        onCreated={reloadProjects}
        onError={setError}
      />
    );
  }

  return (
    <div class="shell">
      <header class="topbar">
        <label class="node-select">
          <span style={{ color: "var(--text-tertiary)" }}>node</span>
          <strong>{node?.name ?? health?.node ?? "…"}</strong>
        </label>
        <span class="topbar__brand">FluxKrea 26</span>

        <select
          class="node-select"
          value={project.id}
          onChange={(event) => openProject((event.target as HTMLSelectElement).value)}
          aria-label="Project"
          title="The project every screen is scoped to"
        >
          {projects.map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.name}
            </option>
          ))}
        </select>

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
        <RailItem label="Masks" active={screen === "masks"} onClick={() => setScreen("masks")}>
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
        {health?.stale && (
          <div class="banner banner--warn">
            ⚠ This daemon started before the code it is running was last changed.
            Restart it, or it will keep behaving like the old version.
          </div>
        )}
        {error && <div class="banner">⚠ {error}</div>}

        {screen === "datasets" &&
          (dataset ? (
            <GalleryScreen
              dataset={dataset}
              onError={setError}
              onOpenReview={() => setScreen("masks")}
            />
          ) : (
            <NoDataset project={project} onAdd={() => setPicking(true)} />
          ))}

        {screen === "masks" &&
          (dataset ? (
            <ReviewScreen dataset={dataset} detectors={detectors} onError={setError} />
          ) : (
            <NoDataset project={project} onAdd={() => setPicking(true)} />
          ))}

        {screen === "training" && (
          <TrainScreen
            datasets={datasets}
            dataset={dataset}
            project={project}
            onDataset={setDataset}
            onProject={(updated) =>
              setProjects((existing) =>
                (existing ?? []).map((entry) => (entry.id === updated.id ? updated : entry)),
              )
            }
            onError={setError}
          />
        )}

        {screen === "settings" && (
          <SettingsScreen
            project={project}
            projects={projects}
            onError={setError}
            onProjectsChanged={reloadProjects}
            onOpenProject={openProject}
            onCloseProject={closeProject}
          />
        )}
      </main>

      {picking && (
        <DatasetPicker
          datasets={datasets}
          onClose={() => setPicking(false)}
          onChanged={(added?: string) => void onDatasetsChanged(added)}
          onRemove={async (id) => {
            // Out of the project, not off the node. On a shared daemon the
            // same folder may be in somebody else's project, and a button
            // in a browser must not be able to take it from them.
            if (projectId) await api.removeProjectDataset(projectId, id);
          }}
        />
      )}
    </div>
  );
}

function NoDataset({ project, onAdd }: { project: Project; onAdd(): void }) {
  return (
    <div class="empty">
      <div class="empty__title">No dataset in {project.name}</div>
      <div>Point this project at a folder of images to get started.</div>
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
    masks: "Masks",
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
