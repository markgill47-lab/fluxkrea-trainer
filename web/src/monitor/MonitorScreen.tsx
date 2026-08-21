/**
 * The training monitor — doc 09.
 *
 * "Stat tiles first because they answer 'is it healthy' in one glance. The
 * chart is the largest single element since it is what gets watched.
 * Samples and logs share the lower band."
 *
 * This is the first screen driven by SSE rather than polling. A training
 * run emits for hours; polling it would either lag or hammer the tunnel.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import { subscribe, type ConnectionState } from "~/api/stream";
import type { Job, LossPayload, SampleImage, StreamEvent } from "~/api/types";
import { LogStream, type LogLine } from "./LogStream";
import { LossChart } from "./LossChart";
import { SampleStrip } from "./SampleStrip";
import { StatTiles } from "./StatTiles";

interface Props {
  onError(message: string | null): void;
  /** Open on this job rather than the newest. Set after submitting one. */
  initialJob?: string | null;
}

/** Lines kept in the browser. The daemon keeps the durable record. */
const MAX_LINES = 20_000;

/** How often to re-pull the derived series while a run is live. */
const LOSS_INTERVAL = 4000;

export function MonitorScreen({ onError, initialJob = null }: Props) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState<string | null>(initialJob);
  const [job, setJob] = useState<Job | null>(null);
  const [loss, setLoss] = useState<LossPayload | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [logScale, setLogScale] = useState(false);
  const [filter, setFilter] = useState("");
  const [minLevel, setMinLevel] = useState<LogLine["level"]>("info");
  const [loading, setLoading] = useState(true);
  const pending = useRef<LogLine[]>([]);
  const onPickOutlier = useCallback(
    (image: string) => onError(`outlier: ${image} — highest mean loss in this run`),
    [onError],
  );

  // -- job list ------------------------------------------------------------

  const reloadJobs = useCallback(async () => {
    try {
      const payload = await api.jobs();
      setJobs(payload.jobs);
      setJobId((current) => current ?? initialJob ?? payload.jobs[0]?.id ?? null);
      onError(null);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [onError, initialJob]);

  useEffect(() => {
    void reloadJobs();
  }, [reloadJobs]);

  // -- the selected job ----------------------------------------------------

  useEffect(() => {
    if (!jobId) return;
    setLines([]);
    pending.current = [];
    setLoss(null);
    void api.job(jobId).then(setJob).catch(() => undefined);
    void api.samples(jobId).then((payload) => setSamples(payload.samples)).catch(() => undefined);
  }, [jobId]);

  // -- the event stream ----------------------------------------------------

  const statusRef = useRef(job?.status);
  statusRef.current = job?.status;

  useEffect(() => {
    if (!jobId) return;

    // Batched into animation frames: a chatty run emits faster than the
    // browser paints, and a setState per line would spend the whole run in
    // reconciliation.
    let frame = 0;
    const flush = () => {
      frame = 0;
      if (!pending.current.length) return;
      const batch = pending.current;
      pending.current = [];
      setLines((current) => {
        const next = current.concat(batch);
        return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
      });
    };

    const stop = subscribe(`/api/v1/jobs/${jobId}/events`, {
      onState: setConnection,
      // The daemon closes the stream when a run ends, and EventSource
      // cannot tell that from a dropped tunnel. Without this a finished
      // run reconnects forever and reads as "retrying".
      isComplete: () =>
        !!statusRef.current && !["queued", "running"].includes(statusRef.current),
      onEvent: (event: StreamEvent) => {
        if (event.kind === "log") {
          pending.current.push({ index: event.index, line: event.line, level: event.level });
          if (!frame) frame = requestAnimationFrame(flush);
        } else if (event.kind === "progress") {
          setJob((current) =>
            current ? { ...current, progress: { step: event.step, total: event.total } } : current,
          );
        } else if (event.kind === "finished") {
          void api.job(jobId).then(setJob).catch(() => undefined);
          void api.samples(jobId).then((p) => setSamples(p.samples)).catch(() => undefined);
          void api.loss(jobId).then(setLoss).catch(() => undefined);
          void reloadJobs();
        }
      },
    });

    return () => {
      stop();
      if (frame) cancelAnimationFrame(frame);
    };
  }, [jobId, reloadJobs]);

  // The derived series is pulled rather than streamed: EMA, trend and
  // outliers are computed over the whole series, so a per-point push would
  // send the same recomputation hundreds of times a minute.
  useEffect(() => {
    if (!jobId) return;
    let live = true;

    const pull = async () => {
      try {
        const payload = await api.loss(jobId);
        if (live) setLoss(payload);
        // Samples arrive on disk as the run writes them, so they are
        // re-listed on the same beat. Fetched once at selection, the strip
        // would freeze at whatever existed when the screen opened.
        const listing = await api.samples(jobId);
        if (live) setSamples(listing.samples);
      } catch {
        /* the stream reports connection trouble; this stays quiet */
      }
    };

    void pull();
    const timer = setInterval(() => {
      if (job?.status === "running") void pull();
    }, LOSS_INTERVAL);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [jobId, job?.status]);

  // -- control -------------------------------------------------------------

  const stop = useCallback(async () => {
    if (!jobId) return;
    try {
      await api.cancelJob(jobId);
      await reloadJobs();
      setJob(await api.job(jobId));
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    }
  }, [jobId, reloadJobs, onError]);

  const running = job?.status === "running";
  const queued = job?.status === "queued";

  const logLines = useMemo(() => lines, [lines]);

  if (loading) {
    return (
      <div class="empty">
        <span class="spinner" />
      </div>
    );
  }

  if (!jobs.length) {
    return (
      <div class="empty">
        <div class="empty__title">No training runs on this node</div>
        <div>
          Submit one with <code class="mono">fk train --model flux2 --dataset poses</code>
        </div>
      </div>
    );
  }

  return (
    <div class="monitor">
      <header class="monitor__head">
        <select
          class="node-select"
          value={jobId ?? ""}
          onChange={(event) => setJobId((event.target as HTMLSelectElement).value)}
          aria-label="Job"
        >
          {jobs.map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.spec.name || entry.spec.model} · {entry.status}
            </option>
          ))}
        </select>

        {job && (
          <>
            <span class="monitor__title">{job.spec.name || job.spec.model}</span>
            <span class={`pill-status pill-status--${job.status}`}>{job.status}</span>
            <span class="hint mono">gpu {job.device}</span>
          </>
        )}

        <span class="topbar__spacer" />

        <span class={`conn conn--${connection === "open" ? "up" : "down"}`}>
          <span class="conn__dot" aria-hidden="true" />
          {connection === "open" ? "streaming" : connection}
        </span>

        <button class="btn" onClick={() => setLogScale((on) => !on)} aria-pressed={logScale}>
          {logScale ? "log" : "linear"}
        </button>
        <button class="btn" onClick={() => void stop()} disabled={!running && !queued}>
          {queued ? "Dequeue" : "Stop"}
        </button>
      </header>

      <StatTiles job={job} loss={loss} />

      <div class="monitor__chart">
        {loss && loss.points.length > 1 ? (
          <LossChart
            series={loss}
            outliers={withSteps(loss)}
            logScale={logScale}
            onPickOutlier={onPickOutlier}
          />
        ) : (
          <div class="empty">
            <div class="empty__title">No loss recorded yet</div>
            <div>Points appear as the run reports them</div>
          </div>
        )}
      </div>

      <SampleStrip samples={samples} />

      <LogStream
        lines={logLines}
        minLevel={minLevel}
        filter={filter}
        onFilter={setFilter}
        onLevel={setMinLevel}
      />
    </div>
  );
}

/**
 * Give each outlier a step to be drawn at.
 *
 * The analytics identify *which image* is an outlier, not when — an image
 * is seen many times across a run. The marker goes at the point where that
 * image's mean sits closest to the curve, which is the most honest single
 * position for something that is really a property of the whole series.
 */
function withSteps(loss: LossPayload) {
  return loss.outliers.map((outlier) => {
    let best = loss.points[0]?.step ?? 0;
    let closest = Infinity;
    for (const point of loss.points) {
      const distance = Math.abs(point.value - outlier.mean);
      if (distance < closest) {
        closest = distance;
        best = point.step;
      }
    }
    return { ...outlier, step: best };
  });
}
