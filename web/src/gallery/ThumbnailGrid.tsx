/**
 * The thumbnail grid: 2D virtualization over a server-thumbnailed dataset.
 *
 * Doc 10 sizes this at 10k+ items, which rules out rendering the lot. The
 * virtualizer windows rows; each row lays out its own cells, so only the
 * visible rectangle exists in the DOM.
 *
 * Cells carry their status as overlays rather than as a separate legend
 * (doc 08): quality, caption-present, mask-present, selection. That is the
 * whole point of a grid over a table here — you are looking for the odd
 * one out, and the odd one out should be visible without reading.
 */

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  Virtualizer,
  elementScroll,
  observeElementOffset,
  observeElementRect,
} from "@tanstack/virtual-core";
import { assets } from "~/api/client";
import type { Item } from "~/api/types";

interface Props {
  dataset: string;
  items: Item[];
  selected: Set<string>;
  focused: string | null;
  cell: number;
  onSelect(stem: string, event: MouseEvent): void;
  onOpen(stem: string): void;
}

/** Gap between cells, in pixels. Matches the 4px spacing step in doc 07. */
const GAP = 8;

export function ThumbnailGrid({
  dataset,
  items,
  selected,
  focused,
  cell,
  onSelect,
  onOpen,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [, force] = useState(0);

  useEffect(() => {
    const host = scrollRef.current;
    if (!host) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const columns = Math.max(1, Math.floor((width + GAP) / (cell + GAP)));
  const rows = Math.ceil(items.length / columns);
  // Cells are square plus a caption strip.
  const rowHeight = cell + 22 + GAP;

  const virtualizer = useMemo(
    () =>
      new Virtualizer<HTMLDivElement, HTMLElement>({
        count: rows,
        getScrollElement: () => scrollRef.current,
        estimateSize: () => rowHeight,
        overscan: 3,
        scrollToFn: elementScroll,
        observeElementRect,
        observeElementOffset,
        onChange: () => force((n) => n + 1),
      }),
    [rows, rowHeight],
  );

  useEffect(() => {
    virtualizer._willUpdate();
    return virtualizer._didMount();
  }, [virtualizer]);

  useEffect(() => {
    virtualizer.setOptions({ ...virtualizer.options, count: rows });
    virtualizer.measure();
  }, [rows, rowHeight, virtualizer]);

  // Keyboard navigation moves focus; the grid has to follow it.
  useEffect(() => {
    if (!focused || !columns) return;
    const index = items.findIndex((item) => item.stem === focused);
    if (index >= 0) virtualizer.scrollToIndex(Math.floor(index / columns), { align: "auto" });
  }, [focused, items, columns, virtualizer]);

  if (!items.length) {
    return (
      <div class="grid" ref={scrollRef}>
        <div class="empty">
          <div class="empty__title">Nothing matches this filter</div>
          <div>Clear the filter chips to see the whole dataset</div>
        </div>
      </div>
    );
  }

  return (
    <div class="grid" ref={scrollRef}>
      <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
        {virtualizer.getVirtualItems().map((virtual) => {
          const start = virtual.index * columns;
          const slice = items.slice(start, start + columns);
          return (
            <div
              key={virtual.key}
              class="grid__row"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                transform: `translateY(${virtual.start}px)`,
                gap: `${GAP}px`,
              }}
            >
              {slice.map((item) => (
                <Cell
                  key={item.stem}
                  dataset={dataset}
                  item={item}
                  size={cell}
                  selected={selected.has(item.stem)}
                  focused={item.stem === focused}
                  onSelect={onSelect}
                  onOpen={onOpen}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Cell({
  dataset,
  item,
  size,
  selected,
  focused,
  onSelect,
  onOpen,
}: {
  dataset: string;
  item: Item;
  size: number;
  selected: boolean;
  focused: boolean;
  onSelect(stem: string, event: MouseEvent): void;
  onOpen(stem: string): void;
}) {
  // 480px thumbnails above this, so a large cell is not an upscaled 160.
  const source = assets.thumb(dataset, item.stem, size > 170 ? 480 : 160, item.token);

  return (
    <figure
      class={`cell${selected ? " cell--selected" : ""}${focused ? " cell--focused" : ""}`}
      style={{ width: `${size}px` }}
      aria-selected={selected}
      onClick={(event) => onSelect(item.stem, event as unknown as MouseEvent)}
      onDblClick={() => onOpen(item.stem)}
      title={item.filename}
    >
      <div class="cell__frame" style={{ height: `${size}px` }}>
        <img class="cell__img" src={source} alt="" loading="lazy" decoding="async" />

        <div class="cell__badges">
          {item.quality && <span class={`badge badge--${item.quality}`}>{item.quality}</span>}
        </div>

        <div class="cell__status">
          <span
            class={`dot${item.has_caption ? " dot--on" : " dot--off"}`}
            title={item.has_caption ? "has a caption" : "no caption"}
          >
            C
          </span>
          <span
            class={`dot${item.has_mask ? " dot--on" : " dot--off"}`}
            title={item.has_mask ? "has a mask" : "no mask"}
          >
            M
          </span>
        </div>

        {selected && <span class="cell__check" aria-hidden="true">✓</span>}
      </div>
      <figcaption class="cell__name">{item.stem}</figcaption>
    </figure>
  );
}
