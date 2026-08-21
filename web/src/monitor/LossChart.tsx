/**
 * The loss chart — uPlot.
 *
 * Doc 10 lists what the chart has to do and then says a heavyweight
 * declarative library will fight every one of them. uPlot is the opposite
 * shape: a canvas renderer you drive, which is why it is here.
 *
 * - **Streaming append** via `setData`, which redraws without rebuilding.
 * - **Decimation** is done server-side (`analytics/loss.py`), so what
 *   arrives is already the right number of points for the pixels.
 * - **Two series plus a secondary axis**: raw loss, EMA, learning rate.
 * - **Log-scale toggle** on the value axis.
 * - **Brush-to-zoom** with a reset, and a hover crosshair.
 * - **Outlier markers** that carry an image id and are clickable — doc 09
 *   calls that link "the most useful thing Klein's analytics produce".
 */

import { useEffect, useLayoutEffect, useRef } from "preact/hooks";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { LossSeries, Outlier } from "~/api/types";

interface Props {
  series: LossSeries;
  outliers: Outlier[];
  logScale: boolean;
  onPickOutlier(imageId: string): void;
}

function token(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function LossChart({ series, outliers, logScale, onPickOutlier }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const outlierRef = useRef<Outlier[]>(outliers);
  outlierRef.current = outliers;
  // Held in a ref, not read from the closure: a parent passing an inline
  // arrow would otherwise change this component's effect identity on every
  // render, tearing down and rebuilding the plot — with empty data — many
  // times a second. The symptom is a chart that stays blank while the data
  // is demonstrably arriving.
  const pickRef = useRef(onPickOutlier);
  pickRef.current = onPickOutlier;

  // Build once. Rebuilding per point would defeat the whole purpose.
  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const accent = token("--accent", "#4A9EFF");
    const ema = token("--state-success", "#3FB950");
    const grid = token("--border", "#333");
    const text = token("--text-secondary", "#A0A0A0");

    const options: uPlot.Options = {
      width: host.clientWidth,
      height: host.clientHeight,
      padding: [12, 16, 0, 0],
      cursor: {
        drag: { x: true, y: false, setScale: true }, // brush-to-zoom
        focus: { prox: 24 },
      },
      scales: {
        x: { time: false },
        y: { distr: logScale ? 3 : 1 },
      },
      axes: [
        {
          stroke: text,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          font: '11px "JetBrains Mono", monospace',
        },
        {
          stroke: text,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          font: '11px "JetBrains Mono", monospace',
          size: 56,
        },
      ],
      series: [
        { label: "step" },
        {
          label: "loss",
          stroke: accent,
          width: 1,
          points: { show: false },
          value: (_self, raw) => (raw == null ? "—" : raw.toFixed(5)),
        },
        {
          label: "ema",
          stroke: ema,
          width: 2,
          points: { show: false },
          value: (_self, raw) => (raw == null ? "—" : raw.toFixed(5)),
        },
      ],
      hooks: {
        // Outlier markers are drawn on top of the series rather than being
        // a fourth series: they are annotations at a step, not a signal
        // sampled at every step, and a sparse series would interpolate.
        draw: [
          (self) => {
            const marks = outlierRef.current;
            if (!marks.length) return;
            const ctx = self.ctx;
            const warn = token("--state-warning", "#D29922");
            ctx.save();
            ctx.fillStyle = warn;
            for (const mark of marks) {
              if (mark.step == null) continue;
              const x = self.valToPos(mark.step, "x", true);
              const y = self.valToPos(mark.mean, "y", true);
              if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
              ctx.beginPath();
              ctx.arc(x, y, 4, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.restore();
          },
        ],
      },
    };

    const plot = new uPlot(options, [[], [], []], host);
    plotRef.current = plot;

    const observer = new ResizeObserver(([entry]) => {
      if (entry) plot.setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(host);

    // Clicking near a marker opens the image it blames.
    const onClick = (event: MouseEvent) => {
      const marks = outlierRef.current;
      if (!marks.length) return;
      const rect = host.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      for (const mark of marks) {
        if (mark.step == null) continue;
        const mx = plot.valToPos(mark.step, "x", true);
        const my = plot.valToPos(mark.mean, "y", true);
        if (Math.hypot(mx - x, my - y) < 10) {
          pickRef.current(mark.image_id);
          return;
        }
      }
    };
    host.addEventListener("click", onClick);

    return () => {
      observer.disconnect();
      host.removeEventListener("click", onClick);
      plot.destroy();
      plotRef.current = null;
    };
  }, [logScale]);

  // Append: setData redraws from the same plot instance.
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;

    const steps = series.points.map((point) => point.step);
    const values = series.points.map((point) => point.value);
    // The EMA arrives on the same decimated steps, so it aligns index for
    // index; anything else would need interpolation and would lie.
    const emaValues = series.ema.length
      ? series.ema.map((point) => point.value)
      : new Array(steps.length).fill(null);

    plot.setData([steps, values, emaValues] as uPlot.AlignedData);
  }, [series]);

  return <div class="chart" ref={hostRef} />;
}
