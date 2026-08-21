/**
 * Stat tiles — doc 09 puts them first because they answer "is it healthy"
 * in one glance.
 *
 * Doc 08: "Fixed width so a row of them does not reflow as values change
 * digits", and tabular numerals throughout, because a number that jitters
 * on every tick is genuinely unpleasant over a long run.
 */

import type { Job, LossPayload } from "~/api/types";

interface Props {
  job: Job | null;
  loss: LossPayload | null;
}

export function StatTiles({ job, loss }: Props) {
  const step = job?.progress.step ?? 0;
  const total = job?.progress.total ?? job?.spec.steps ?? 0;
  const trend = loss?.trend;

  return (
    <div class="tiles">
      <Tile label="step" value={step ? step.toLocaleString() : "—"} sub={total ? `/ ${total.toLocaleString()}` : ""} />
      <Tile label="loss" value={loss?.latest != null ? loss.latest.toFixed(4) : "—"}>
        {loss && loss.points.length > 1 && <Sparkline points={loss.points.map((p) => p.value)} />}
      </Tile>
      <Tile
        label={`ema ${loss?.ema_window ?? 50}`}
        value={loss?.latest_ema != null ? loss.latest_ema.toFixed(4) : "—"}
        sub={trend ? `${arrowFor(trend.status)} ${trend.status}` : ""}
        tone={trend?.status === "degrading" ? "warn" : undefined}
      />
      <Tile label="eta" value={etaFor(job, step, total)} />
      {/* "0 outliers" is a claim about the images, and it needs the backend
        * to have said which image each step used. ai-toolkit does not, so
        * a confident zero there was answering a question the data could
        * not reach. An em dash and the reason instead. */}
      <Tile
        label="outliers"
        value={loss?.attributed ? String(loss.outliers.length) : "—"}
        sub={
          loss && !loss.attributed
            ? "no per-image loss from this trainer"
            : (loss?.outliers[0]?.image_id ?? "")
        }
        tone={loss?.attributed && loss.outliers.length ? "warn" : undefined}
      />
    </div>
  );
}

function arrowFor(status: string): string {
  return { improving: "↘", degrading: "↗", stable: "→", converged: "=", unknown: "?" }[status] ?? "";
}

/**
 * Elapsed time extrapolated to the remaining steps.
 *
 * Deliberately naive: a rate averaged over the whole run so far. A
 * window-based estimate looks more precise and is more wrong, because
 * caching and sampling make the early steps unrepresentative in a way that
 * a short window over-corrects for.
 */
function etaFor(job: Job | null, step: number, total: number): string {
  if (!job?.started || !step || !total || step >= total) return "—";
  const elapsed = Date.now() / 1000 - job.started;
  const remaining = (elapsed / step) * (total - step);
  if (!Number.isFinite(remaining) || remaining <= 0) return "—";

  const hours = Math.floor(remaining / 3600);
  const minutes = Math.floor((remaining % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  // Seconds below a minute: "0m" on a run that has 40 seconds left reads
  // as broken rather than as nearly finished.
  if (minutes) return `${minutes}m`;
  return `${Math.max(1, Math.round(remaining))}s`;
}

function Tile({
  label,
  value,
  sub,
  tone,
  children,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "warn";
  children?: preact.ComponentChildren;
}) {
  return (
    <div class={`tile${tone ? ` tile--${tone}` : ""}`}>
      <span class="tile__label">{label}</span>
      <span class="tile__value tabular">{value}</span>
      {sub ? <span class="tile__sub tabular">{sub}</span> : null}
      {children}
    </div>
  );
}

/** 60x16, no axes, single series — doc 08. Inline SVG, no library. */
function Sparkline({ points }: { points: number[] }) {
  const sample = points.slice(-60);
  const min = Math.min(...sample);
  const max = Math.max(...sample);
  const span = max - min || 1;
  const path = sample
    .map((value, index) => {
      const x = (index / Math.max(1, sample.length - 1)) * 60;
      const y = 16 - ((value - min) / span) * 16;
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg class="tile__spark" width="60" height="16" viewBox="0 0 60 16" aria-hidden="true">
      <path d={path} fill="none" stroke="var(--accent)" stroke-width="1" />
    </svg>
  );
}
