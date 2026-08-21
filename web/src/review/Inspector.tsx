/**
 * The inspector: boxes, mask parameters, detector.
 *
 * The box list is not decoration — doc 10 makes it the canvas's accessible
 * equivalent. The canvas cannot be reached by a screen reader, so every
 * box is a focusable item with its geometry as text, and every canvas
 * operation has a keyboard path.
 */

import type { Box } from "~/api/types";
import { MANUAL } from "~/lib/boxes";
import type { MaskSettings } from "./Viewport";

interface Props {
  boxes: Box[];
  selected: number[];
  mask: MaskSettings;
  detectors: string[];
  detector: string;
  detecting: boolean;
  onSelect(indices: number[]): void;
  onMask(next: Partial<MaskSettings>): void;
  onDetector(name: string): void;
  onRedetect(): void;
  onDelete(): void;
}

export function Inspector({
  boxes,
  selected,
  mask,
  detectors,
  detector,
  detecting,
  onSelect,
  onMask,
  onDetector,
  onRedetect,
  onDelete,
}: Props) {
  const detected = boxes.filter((box) => box.src !== MANUAL).length;
  const manual = boxes.length - detected;
  const one = selected.length === 1 ? boxes[selected[0]!] : null;

  return (
    <aside class="inspector" aria-label="Inspector">
      <section class="inspector__section">
        <h2 class="inspector__label">Boxes</h2>
        <div class="kv">
          <span class="kv__key">detected</span>
          <span class="kv__value">{detected}</span>
        </div>
        <div class="kv">
          <span class="kv__key">manual</span>
          <span class="kv__value">{manual}</span>
        </div>

        {boxes.length === 0 ? (
          <p style={{ margin: "8px 0 0", color: "var(--state-warning)", fontSize: "12px" }}>
            No boxes. Press <kbd>B</kbd> and drag to draw one.
          </p>
        ) : (
          <ul class="box-list" style={{ marginTop: "8px" }}>
            {boxes.map((box, index) => (
              <li key={index}>
                <button
                  class="box-list__item"
                  aria-selected={selected.includes(index)}
                  onClick={() => onSelect([index])}
                >
                  <span
                    class="box-list__swatch"
                    style={{
                      background:
                        box.src === MANUAL ? "var(--box-manual)" : "var(--box-detected)",
                    }}
                    aria-hidden="true"
                  />
                  <span>
                    {box.src}
                    {box.conf != null ? ` ${box.conf.toFixed(2)}` : ""}
                  </span>
                  <span class="box-list__geometry mono">
                    {box.w}×{box.h}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {one && (
          <div style={{ marginTop: "10px" }}>
            <div class="kv">
              <span class="kv__key">x</span>
              <span class="kv__value mono">{one.x}</span>
            </div>
            <div class="kv">
              <span class="kv__key">y</span>
              <span class="kv__value mono">{one.y}</span>
            </div>
            <div class="kv">
              <span class="kv__key">w</span>
              <span class="kv__value mono">{one.w}</span>
            </div>
            <div class="kv">
              <span class="kv__key">h</span>
              <span class="kv__value mono">{one.h}</span>
            </div>
            <button class="btn" style={{ marginTop: "8px" }} onClick={onDelete}>
              Delete box
            </button>
          </div>
        )}
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Mask</h2>

        <Slider
          label="expand"
          value={mask.expand}
          min={1}
          max={3}
          step={0.05}
          format={(v) => `${v.toFixed(2)}×`}
          onChange={(expand) => onMask({ expand })}
        />
        <Slider
          label="up bias"
          value={mask.expandUp}
          min={1}
          max={2.5}
          step={0.05}
          format={(v) => `${v.toFixed(2)}×`}
          onChange={(expandUp) => onMask({ expandUp })}
        />
        <Slider
          label="feather"
          value={mask.feather}
          min={0}
          max={48}
          step={1}
          format={(v) => `${v}px`}
          onChange={(feather) => onMask({ feather })}
        />
        <Slider
          label="opacity"
          value={mask.opacity}
          min={0}
          max={1}
          step={0.05}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(opacity) => onMask({ opacity })}
        />

        <p style={{ margin: "6px 0 0", color: "var(--text-tertiary)", fontSize: "11px" }}>
          Preview only. The exported mask is rendered by the daemon from the
          same boxes and settings.
        </p>
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Detector</h2>
        <div class="field">
          <select
            value={detector}
            onChange={(event) => onDetector((event.target as HTMLSelectElement).value)}
          >
            {detectors.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <button class="btn" onClick={onRedetect} disabled={detecting}>
          {detecting ? <span class="spinner" /> : null}
          {detecting ? "Detecting…" : "Re-detect this image"}
        </button>
        <p style={{ margin: "6px 0 0", color: "var(--text-tertiary)", fontSize: "11px" }}>
          Manual boxes are kept.
        </p>
      </section>
    </aside>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format(value: number): string;
  onChange(value: number): void;
}) {
  return (
    <div class="field">
      <span class="field__label">{label}</span>
      <span class="field__value">{format(value)}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onInput={(event) => onChange(Number((event.target as HTMLInputElement).value))}
      />
    </div>
  );
}
