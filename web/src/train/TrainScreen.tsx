/**
 * The training tab: configure a run, or watch the one that is going.
 *
 * Two views rather than one scrolling page, because they are used at
 * different times and the monitor wants the whole viewport. Which one
 * opens is decided by what the node is doing — if something is running you
 * almost certainly came here to look at it, and if nothing is you came to
 * start one. The choice is then yours; it is not re-decided under you when
 * a run happens to finish.
 */

import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Dataset, Job, ModelInfo } from "~/api/types";
import { MonitorScreen } from "~/monitor/MonitorScreen";
import { TrainForm } from "./TrainForm";

type View = "configure" | "monitor";

/** How often to re-check whether a run is going. */
const POLL = 4000;

export function TrainScreen({
  datasets,
  onError,
}: {
  datasets: Dataset[];
  onError(message: string | null): void;
}) {
  const [view, setView] = useState<View | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [devices, setDevices] = useState(1);
  const [active, setActive] = useState<Job | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const chosen = useRef(false);

  const poll = useCallback(async () => {
    try {
      const payload = await api.jobs();
      const running =
        payload.jobs.find((job) => job.status === "running") ??
        payload.jobs.find((job) => job.status === "queued") ??
        null;
      setActive(running);
      setDevices(payload.devices);

      // Only the first answer picks the view. After that the user owns it.
      if (!chosen.current) {
        chosen.current = true;
        setView(running ? "monitor" : "configure");
      }
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    }
  }, [onError]);

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

        {active ? (
          <span class="train-screen__active">
            <span class="conn__dot" style={{ background: "var(--state-running, var(--accent))" }} />
            {active.status} · {active.spec.name || active.id}
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
          <TrainForm
            datasets={datasets}
            models={models}
            devices={devices}
            locked={active !== null}
            onSubmitted={onSubmitted}
            onError={onError}
          />
        ) : (
          <MonitorScreen onError={onError} initialJob={focus ?? active?.id ?? null} />
        )}
      </div>
    </div>
  );
}
