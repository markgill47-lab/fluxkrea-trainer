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
import { api, ApiError, assets, isAbort } from "~/api/client";
import { subscribe, type ConnectionState } from "~/api/stream";
import type {
  Artifact,
  ArtifactsResponse,
  Job,
  LossPayload,
  SampleImage,
  StreamEvent,
} from "~/api/types";
import { LogStream, type LogLine } from "./LogStream";
import { LossChart } from "./LossChart";
import { SampleStrip } from "./SampleStrip";
import { StatTiles } from "./StatTiles";

interface Props {
  onError(message: string | null): void;
  /** Open on this job rather than the newest. Set after submitting one. */
  initialJob?: string | null;
  /** Show only this project's runs. On a shared node the unfiltered list is
   *  everybody's, and other people's runs are not yours to watch. */
  project?: string;
}

/** Lines kept in the browser. The daemon keeps the durable record. */
const MAX_LINES = 20_000;

/** How often to re-pull the derived series while a run is live. */
const LOSS_INTERVAL = 4000;

export function MonitorScreen({ onError, initialJob = null, project }: Props) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState<string | null>(initialJob);
  const [job, setJob] = useState<Job | null>(null);
  const [loss, setLoss] = useState<LossPayload | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactsResponse | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [published, setPublished] = useState<string | null>(null);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  /**
   * Whose runs to list.
   *
   * The project by default: on a shared node the unfiltered list is
   * everybody's, and scrolling past nine other people's runs to find your
   * own is not a monitor. But "all" has to exist — runs submitted from the
   * CLI and from the fleet carry no project, and without this they would
   * be invisible in the UI on the very node that produced them.
   */
  const [scope, setScope] = useState<"project" | "node">(project ? "project" : "node");
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
      const payload = await api.jobs(scope === "project" ? project : undefined);
      setJobs(payload.jobs);
      setJobId((current) => current ?? initialJob ?? payload.jobs[0]?.id ?? null);
      onError(null);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [onError, initialJob, project, scope]);

  useEffect(() => {
    void reloadJobs();
  }, [reloadJobs]);

  // Changing scope can drop the selected run out of the list. Clear it and
  // let the reload pick the first one rather than streaming a job the
  // picker no longer shows.
  useEffect(() => {
    setJobId((current) => (current && jobs.some((entry) => entry.id === current) ? current : null));
  }, [jobs]);

  // -- the selected job ----------------------------------------------------

  useEffect(() => {
    if (!jobId) return;
    setLines([]);
    pending.current = [];
    setLoss(null);
    setArtifacts(null);
    setPublished(null);
    void api.job(jobId).then(setJob).catch(() => undefined);
    void api.samples(jobId).then((payload) => setSamples(payload.samples)).catch(() => undefined);
  }, [jobId]);

  /**
   * What this run produced, once it has stopped producing it.
   *
   * Only listed for a run that has finished. Mid-run the output folder
   * holds rotated checkpoints that are about to be deleted, and offering a
   * download of a file that will not exist by the time the click lands is
   * worse than offering nothing.
   */
  const status = job?.status;
  useEffect(() => {
    if (!jobId || !status || ["queued", "running"].includes(status)) return;
    let live = true;
    void api
      .artifacts(jobId)
      .then((payload) => {
        if (live) setArtifacts(payload);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [jobId, status]);

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

  const publish = useCallback(
    async (artifact: Artifact) => {
      if (!jobId) return;
      setPublishing(true);
      try {
        const result = await api.publishArtifact(jobId, { artifact: artifact.name });
        setPublished(result.published);
        onError(null);
      } catch (error) {
        if (isAbort(error)) return;
        // A name collision is the expected refusal on a shared node - the
        // file already there is somebody else's afternoon. The message says
        // so; nothing is retried automatically.
        onError(error instanceof ApiError ? error.message : String(error));
      } finally {
        setPublishing(false);
      }
    },
    [jobId, onError],
  );

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
        <div class="empty__title">
          {scope === "project" ? "No runs in this project yet" : "No training runs on this node"}
        </div>
        <div>
          Configure one on the other tab, or{" "}
          <code class="mono">fk train --model flux2 --dataset poses</code>
        </div>
        {scope === "project" && (
          <div>
            {/* The node may well have runs — somebody else's, or one
                submitted from the CLI with no project attached. Saying
                "none on this node" when there are nine would be a lie. */}
            <button class="btn" onClick={() => setScope("node")}>
              Show every run on this node
            </button>
          </div>
        )}
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

        {project && (
          <button
            class="btn"
            onClick={() => setScope((current) => (current === "project" ? "node" : "project"))}
            title="Switch between this project's runs and every run on the node"
          >
            {scope === "project" ? "this project" : "whole node"}
          </button>
        )}

        {job && (
          <>
            <span class="monitor__title">{job.spec.name || job.spec.model}</span>
            {scope === "node" && job.project && (
              <span class="chip">{job.project}</span>
            )}
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

      <SampleStrip samples={samples} jobId={jobId} />

      {artifacts && artifacts.artifacts.length > 0 && jobId && (
        <ArtifactBar
          jobId={jobId}
          payload={artifacts}
          publishing={publishing}
          published={published}
          onPublish={publish}
        />
      )}

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
 * What the run produced, and the two things anybody wants to do with it.
 *
 * **Download** is a plain link rather than a fetch: these are 90MB and up,
 * and pulling one through JavaScript into a blob would hold the whole file
 * in memory and lose the progress the browser gives for free.
 *
 * **Publish** copies it into ComfyUI's `models/loras/<family>` *on the
 * node*. Over a LAN URL that is a different machine from the one looking
 * at this, which is the whole reason it is a button rather than an
 * instruction. The family comes off the model record, so nobody chooses
 * between `flux2` and `krea2` and nobody gets it wrong.
 */
function ArtifactBar({
  jobId,
  payload,
  publishing,
  published,
  onPublish,
}: {
  jobId: string;
  payload: ArtifactsResponse;
  publishing: boolean;
  published: string | null;
  onPublish(artifact: Artifact): void;
}) {
  const [chosen, setChosen] = useState(payload.artifacts[0]?.name ?? "");
  const artifact =
    payload.artifacts.find((entry) => entry.name === chosen) ?? payload.artifacts[0]!;

  return (
    <section class="artifacts">
      <span class="artifacts__title">Trained LoRA</span>

      {payload.artifacts.length > 1 ? (
        <select
          class="node-select"
          value={artifact.name}
          onChange={(event) => setChosen((event.target as HTMLSelectElement).value)}
          aria-label="Which checkpoint"
        >
          {payload.artifacts.map((entry) => (
            <option key={entry.name} value={entry.name}>
              {entry.final ? "final" : `step ${entry.step?.toLocaleString()}`} ·{" "}
              {megabytes(entry.size)}
            </option>
          ))}
        </select>
      ) : (
        <span class="artifacts__name mono">
          {artifact.name} · {megabytes(artifact.size)}
        </span>
      )}

      <span class="topbar__spacer" />

      {published ? (
        <span class="chip" title={published}>
          ✓ published to {payload.family}
        </span>
      ) : (
        <span class="hint">
          {payload.publishable
            ? `goes to models/loras/${payload.family}`
            : "set backends.comfyui_path on this node to publish"}
        </span>
      )}

      <a class="btn" href={assets.artifact(jobId, artifact.name)} download={artifact.name}>
        Download
      </a>
      <button
        class="btn btn--accent"
        disabled={!payload.publishable || publishing}
        onClick={() => onPublish(artifact)}
        title={
          payload.publishable
            ? `Copy into ${payload.comfyui}/models/loras/${payload.family} on this node`
            : "This node has no backends.comfyui_path configured"
        }
      >
        {publishing ? <span class="spinner" /> : null}
        Publish to ComfyUI
      </button>
    </section>
  );
}

function megabytes(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(0)} MB`;
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
