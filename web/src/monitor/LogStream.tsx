/**
 * The log stream — the hardest of doc 10's three virtualized surfaces.
 *
 * "Follow-tail plus scroll-anchoring plus a text filter that changes row
 * heights." Each is easy alone; together they conflict, and the conflicts
 * are what this file is about:
 *
 * - **Follow-tail auto-disables on a manual scroll up.** Nothing is more
 *   irritating than being yanked back to the bottom while reading a
 *   traceback. It re-enables when you return to the bottom yourself, or
 *   via the jump affordance.
 * - **A filter changes which rows exist**, so the virtualizer's item count
 *   changes under it. Filtering is applied before virtualization and the
 *   measurement cache is invalidated, rather than trying to keep a
 *   scroll position that no longer refers to anything.
 * - **Rows are one line each**, scrolled horizontally rather than wrapped.
 *   Doc 10 anticipated variable heights; measuring them turned out to be
 *   the wrong trade. A wrapped row's height depends on the container
 *   width, so every filter change and every resize invalidates the whole
 *   measurement cache, and the failure mode is rows drawn on top of each
 *   other. Uniform rows make follow-tail and anchoring exact, and no-wrap
 *   is what every log viewer does anyway. The filter still changes which
 *   rows exist, which is the part that mattered.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  Virtualizer,
  elementScroll,
  observeElementOffset,
  observeElementRect,
} from "@tanstack/virtual-core";

export interface LogLine {
  index: number;
  line: string;
  level: "debug" | "info" | "warning" | "error";
}

interface Props {
  lines: LogLine[];
  /** Levels below this are hidden. */
  minLevel: LogLine["level"];
  filter: string;
  onFilter(text: string): void;
  onLevel(level: LogLine["level"]): void;
}

const LEVELS: LogLine["level"][] = ["debug", "info", "warning", "error"];

/** Within this many pixels of the bottom counts as "at the bottom". */
const BOTTOM_SLOP = 24;

/** One line, monospace, 11px/1.45 plus padding. Kept in step with the CSS. */
const ROW_HEIGHT = 18;

export function LogStream({ lines, minLevel, filter, onFilter, onLevel }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const [, force] = useState(0);
  const programmatic = useRef(false);

  const visible = useMemo(() => {
    const floor = LEVELS.indexOf(minLevel);
    const needle = filter.trim().toLowerCase();
    return lines.filter(
      (entry) =>
        LEVELS.indexOf(entry.level) >= floor &&
        (!needle || entry.line.toLowerCase().includes(needle)),
    );
  }, [lines, minLevel, filter]);

  const virtualizer = useMemo(
    () =>
      new Virtualizer<HTMLDivElement, HTMLElement>({
        count: visible.length,
        getScrollElement: () => scrollRef.current,
        estimateSize: () => ROW_HEIGHT,
        overscan: 24,
        scrollToFn: elementScroll,
        observeElementRect,
        observeElementOffset,
        onChange: () => force((n) => n + 1),
      }),
    [],
  );

  useEffect(() => {
    virtualizer._willUpdate();
    return virtualizer._didMount();
  }, [virtualizer]);

  // The filter changes which rows exist, so the count changes under the
  // virtualizer. Row heights do not, which is the point of uniform rows.
  useLayoutEffect(() => {
    virtualizer.setOptions({ ...virtualizer.options, count: visible.length });
    virtualizer.measure();
  }, [visible.length, virtualizer]);

  // Follow-tail.
  useLayoutEffect(() => {
    if (!follow || !visible.length) return;
    programmatic.current = true;
    virtualizer.scrollToIndex(visible.length - 1, { align: "end" });
    // The flag is cleared on the next frame, after the scroll event it
    // caused has been and gone.
    requestAnimationFrame(() => {
      programmatic.current = false;
    });
  }, [visible.length, follow, virtualizer]);

  const onScroll = () => {
    // A scroll we caused must not switch following off.
    if (programmatic.current) return;
    const host = scrollRef.current;
    if (!host) return;
    const atBottom = host.scrollHeight - host.scrollTop - host.clientHeight < BOTTOM_SLOP;
    setFollow(atBottom);
  };

  const jump = () => {
    setFollow(true);
    if (visible.length) virtualizer.scrollToIndex(visible.length - 1, { align: "end" });
  };

  return (
    <section class="logs" aria-label="Log stream">
      <header class="logs__head">
        <input
          class="logs__filter"
          type="search"
          placeholder="filter…"
          value={filter}
          onInput={(event) => onFilter((event.target as HTMLInputElement).value)}
        />
        <select
          class="logs__level"
          value={minLevel}
          onChange={(event) => onLevel((event.target as HTMLSelectElement).value as LogLine["level"])}
          aria-label="Minimum level"
        >
          {LEVELS.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>
        <span class="hint tabular">
          {visible.length}
          {visible.length !== lines.length ? ` / ${lines.length}` : ""}
        </span>
        {!follow && (
          <button class="btn btn--ghost logs__jump" onClick={jump}>
            ↓ latest
          </button>
        )}
      </header>

      <div class="logs__scroll" ref={scrollRef} onScroll={onScroll}>
        <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
          {virtualizer.getVirtualItems().map((virtual) => {
            const entry = visible[virtual.index];
            if (!entry) return null;
            return (
              <div
                key={virtual.key}
                class={`logs__line logs__line--${entry.level}`}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  height: `${ROW_HEIGHT}px`,
                  transform: `translateY(${virtual.start}px)`,
                }}
              >
                {entry.line}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
