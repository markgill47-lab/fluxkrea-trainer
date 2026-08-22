/**
 * The training tab: configure a run, watch one, or see who is in the queue.
 *
 * Two views rather than one scrolling page, because they are used at
 * different times and the monitor wants the whole viewport. Which one
 * opens is decided by what the node is doing — if something is running you
 * almost certainly came here to look at it, and if nothing is you came to
 * start one. The choice is then yours; it is not re-decided under you when
 * a run happens to finish.
 *
 * **The form's state lives in the project.** It used to live in
 * sessionStorage, because switching views unmounts the form and when the
 * form owned its own state that silently reset every field — including the
 * dataset, which fell back to the first registered one. A run then trained
 * the wrong images and said nothing, which is the most expensive kind of
 * bug this screen can have. The project holds it now for the same reason
 * and one more: it is shared, so the three datasets in a project train
 * with the same settings and the second student to sit down inherits them
 * rather than starting from the defaults.
 *
 * **The form does not lock while a run is going.** It used to, on the
 * grounds that a field accepting an edit that changes nothing is a lie
 * about what the run is doing. That was right when the node ran one job
 * for one operator. With a shared queue it is wrong: locking the form
 * while somebody else trains is exactly the thing a queue exists to
 * prevent, and it would make the node first-come-first-served-for-the-day.
 */

import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Dataset, Job, ModelInfo, Project, QueueEntry } from "~/api/types";
import { MonitorScreen } from "~/monitor/MonitorScreen";
import { DEFAULTS, type FormState, TrainForm } from "./TrainForm";

type View = "configure" | "monitor";

/** How often to re-check the queue. */
const POLL = 4000;

/**
 * How long to wait after the last keystroke before writing the shared
 * config back. Long enough that typing a learning rate is one request
 * rather than eight, short enough that switching tabs does not lose it.
 */
const SAVE_DELAY = 600;

/** Read a project's stored form, filling in anything it predates. */
function restore(project: Project): FormState {
  // Spread over the defaults so a field added since this was written
  // arrives with its default rather than as undefined.
  return { ...DEFAULTS, ...(project.config as Partial<FormState>) };
}

export function TrainScreen({
  datasets,
  dataset,
  project,
  onDataset,
  onProject,
  onError,
}: {
  datasets: Dataset[];
  /** The shell's current dataset. One selection for the whole app. */
  dataset: string | null;
  /** The open project: the shared config, and the run's identity in the queue. */
  project: Project;
  onDataset(id: string): void;
  /** Hand the saved project back so the shell's copy does not go stale. */
  onProject(project: Project): void;
  onError(message: string | null): void;
}) {
  const [view, setView] = useState<View | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [devices, setDevices] = useState(1);
  const [active, setActive] = useState<Job | null>(null);
  const [queue, setQueue] = useState<QueueEntry[]>([]);
  const [mine, setMine] = useState<Job[]>([]);
  const [focus, setFocus] = useState<string | null>(null);
  const [form, setFormState] = useState<FormState>(() => restore(project));
  const chosen = useRef(false);
  const saveTimer = useRef<number | undefined>(undefined);
  // The edit a pending timer is going to write, and which project it
  // belongs to. Held so the timer can be flushed rather than merely
  // cancelled — see the unmount effect below.
  const pending = useRef<{ project: string; form: FormState } | null>(null);

  /** Write any pending edit now, rather than when the timer says so. */
  const flush = useCallback(() => {
    window.clearTimeout(saveTimer.current);
    const outstanding = pending.current;
    pending.current = null;
    if (!outstanding) return;
    void api
      .saveProjectConfig(
        outstanding.project,
        outstanding.form as unknown as Record<string, unknown>,
      )
      .then(onProject)
      .catch(() => undefined);
  }, [onProject]);

  // Follow the project when it changes. Without this, switching projects in
  // the top bar would leave the previous one's settings on screen and then
  // save them over the new project's the moment anything was typed.
  useEffect(() => {
    // Whatever is pending belongs to the project being left, so write it
    // before loading the new one over the top of it.
    flush();
    setFormState(restore(project));
  }, [project.id]);

  const setForm = useCallback(
    (update: (current: FormState) => FormState) => {
      setFormState((current) => {
        const next = update(current);
        pending.current = { project: project.id, form: next };
        window.clearTimeout(saveTimer.current);
        // A failed save is not worth interrupting somebody mid-form over:
        // the value is on screen and the next keystroke retries.
        saveTimer.current = window.setTimeout(flush, SAVE_DELAY);
        return next;
      });
    },
    [project.id, flush],
  );

  /**
   * Write the pending edit on the way out — **flush, not cancel**.
   *
   * Switching rail tabs unmounts this screen, and an edit made inside the
   * debounce window would otherwise be dropped on the floor. That is the
   * same shape as the bug this screen's tests were written for: the form
   * silently reverting to settings nobody chose, and a run then training
   * against them.
   */
  useEffect(() => flush, [flush]);

  // The dataset is the shell's, not the form's. Mirrored into the form so
  // `toSpec` has one place to read it from, and so a dataset that vanishes
  // from the registry does not leave a stale id behind.
  useEffect(() => {
    setFormState((current) =>
      current.dataset === (dataset ?? "") ? current : { ...current, dataset: dataset ?? "" },
    );
  }, [dataset]);

  const poll = useCallback(async () => {
    try {
      const payload = await api.jobs(project.id);
      const running =
        payload.jobs.find((job) => job.status === "running") ??
        payload.jobs.find((job) => job.status === "queued") ??
        null;
      setActive(running);
      setMine(payload.jobs);
      setQueue(payload.queue);
      setDevices(payload.devices);

      // Only the first answer picks the view. After that the user owns it.
      if (!chosen.current) {
        chosen.current = true;
        setView(running ? "monitor" : "configure");
      }
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    }
  }, [project.id, onError]);

  useEffect(() => {
    void poll();
    const timer = setInterval(poll, POLL);
    return () => clearInterval(timer);
  }, [poll]);

  useEffect(() => {
    (async () => {
      try {
        setModels((await api.models()).models);
      } catch (error) {
        if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
      }
    })();
  }, [onError]);

  const onSubmitted = useCallback(
    (jobId: string) => {
      setFocus(jobId);
      setView("monitor");
      void poll();
    },
    [poll],
  );

  if (view === null) {
    return (
      <div class="empty">
        <span class="spinner" aria-label="Loading" />
      </div>
    );
  }

  // Runs from other projects that are ahead of ours. The number a student
  // is actually asking, and the reason the queue is not filtered.
  const others = queue.filter((entry) => entry.project !== project.id).length;

  return (
    <div class="train-screen">
      <header class="train-screen__head">
        <div class="segmented" role="tablist" aria-label="Training">
          <button
            role="tab"
            class="segmented__item"
            aria-selected={view === "configure"}
            onClick={() => setView("configure")}
          >
            Configure
          </button>
          <button
            role="tab"
            class="segmented__item"
            aria-selected={view === "monitor"}
            onClick={() => setView("monitor")}
          >
            Monitor
          </button>
        </div>

        <span class="topbar__spacer" />

        {queue.length > 0 && (
          <span class="chip" title="Every run waiting on this node, not just yours">
            {queue.length} queued{others > 0 ? ` · ${others} from other projects` : ""}
          </span>
        )}

        {active ? (
          <span class="train-screen__active">
            <span class="conn__dot" style={{ background: "var(--state-running, var(--accent))" }} />
            {active.status} · {active.spec.name || active.id}
            {active.status === "queued" && (active.position ?? -1) >= 0 && (
              <span class="tabular">
                {" "}
                {active.position === 0 ? "next up" : `${active.position} ahead`}
              </span>
            )}
            {active.progress.total > 0 && (
              <span class="tabular">
                {" "}
                {active.progress.step.toLocaleString()} /{" "}
                {active.progress.total.toLocaleString()}
              </span>
            )}
          </span>
        ) : (
          <span class="train-screen__active train-screen__active--idle">nothing running</span>
        )}
      </header>

      <div class="train-screen__body">
        {view === "configure" ? (
          <>
            <QueuePanel queue={queue} project={project} devices={devices} />
            <TrainForm
              datasets={datasets}
              models={models}
              devices={devices}
              project={project.id}
              queued={queue.length}
              form={form}
              setForm={setForm}
              onDataset={onDataset}
              onSubmitted={onSubmitted}
              onError={onError}
            />
          </>
        ) : (
          <MonitorScreen
            onError={onError}
            project={project.id}
            initialJob={focus ?? active?.id ?? mine[0]?.id ?? null}
          />
        )}
      </div>
    </div>
  );
}

/**
 * Who is waiting, in the order the node will start them.
 *
 * Shown before the form rather than after it, because it is the thing that
 * decides whether pressing Start means "in ten minutes" or "after lunch",
 * and that belongs before the decision rather than after it.
 */
function QueuePanel({
  queue,
  project,
  devices,
}: {
  queue: QueueEntry[];
  project: Project;
  devices: number;
}) {
  if (queue.length === 0) return null;
  return (
    <section class="panel">
      <h2 class="panel__title">Queue</h2>
      <p class="panel__note">
        {devices === 1
          ? "One run at a time on this node."
          : `${devices} runs at a time on this node.`}{" "}
        Each project's next run goes before any project's one after that, so a
        batch from one person does not hold the card all day.
      </p>
      <ol class="queue">
        {queue.map((entry, index) => (
          <li
            key={entry.id}
            class={`queue__row${entry.project === project.id ? " queue__row--mine" : ""}`}
          >
            <span class="queue__position tabular">{index + 1}</span>
            <span class="queue__project">{entry.project || "unclaimed"}</span>
            <span class="queue__name mono">{entry.name || entry.id}</span>
            <span class="queue__model">{entry.model}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
