/**
 * The filmstrip: virtualized item list, zero-detection items pinned first.
 *
 * Doc 09's brief took a position here worth preserving: a sort order was
 * not loud enough. Images with no detections are a *pinned amber group* at
 * the top with a count header, because a miss is the failure mode that
 * defeats the whole feature and the interface should push them at you
 * rather than wait to be asked.
 */

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { Virtualizer, elementScroll, observeElementOffset, observeElementRect } from "@tanstack/virtual-core";
import { assets } from "~/api/client";
import type { Item } from "~/api/types";

interface Props {
  /** Hidden below the minimum viewport; the image keeps the space. */
  hidden?: boolean;
  dataset: string;
  items: Item[];
  selected: string | null;
  onSelect(stem: string): void;
}

const ROW_HEIGHT = 57;

interface Row {
  kind: "header" | "item";
  label?: string;
  count?: number;
  warn?: boolean;
  item?: Item;
}

/**
 * Group the items: no-detections first, then unreviewed, then reviewed.
 *
 * Headers are rows in the same list rather than separate elements, so the
 * virtualizer measures one flat sequence and keyboard navigation does not
 * have to know groups exist.
 */
export function groupItems(items: Item[]): Row[] {
  const empty = items.filter((item) => item.boxes === 0);
  const unreviewed = items.filter((item) => item.boxes > 0 && !item.reviewed);
  const reviewed = items.filter((item) => item.boxes > 0 && item.reviewed);

  const rows: Row[] = [];
  if (empty.length) {
    rows.push({ kind: "header", label: "No detections", count: empty.length, warn: true });
    for (const item of empty) rows.push({ kind: "item", item });
  }
  if (unreviewed.length) {
    rows.push({ kind: "header", label: "Unreviewed", count: unreviewed.length });
    for (const item of unreviewed) rows.push({ kind: "item", item });
  }
  if (reviewed.length) {
    rows.push({ kind: "header", label: "Reviewed", count: reviewed.length });
    for (const item of reviewed) rows.push({ kind: "item", item });
  }
  return rows;
}

export function Filmstrip({ hidden, dataset, items, selected, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const rows = useMemo(() => groupItems(items), [items]);
  const [, force] = useState(0);

  const virtualizer = useMemo(
    () =>
      new Virtualizer<HTMLDivElement, HTMLElement>({
        count: rows.length,
        getScrollElement: () => scrollRef.current,
        estimateSize: (index) => (rows[index]?.kind === "header" ? 25 : ROW_HEIGHT),
        overscan: 8,
        scrollToFn: elementScroll,
        observeElementRect,
        observeElementOffset,
        onChange: () => force((n) => n + 1),
      }),
    [rows],
  );

  useEffect(() => {
    virtualizer._willUpdate();
    return virtualizer._didMount();
  }, [virtualizer]);

  useEffect(() => {
    virtualizer.setOptions({ ...virtualizer.options, count: rows.length });
    virtualizer.measure();
  }, [rows.length, virtualizer]);

  // Keyboard navigation moves the selection; the list has to follow it.
  useEffect(() => {
    if (!selected) return;
    const index = rows.findIndex((row) => row.item?.stem === selected);
    if (index >= 0) virtualizer.scrollToIndex(index, { align: "auto" });
  }, [selected, rows, virtualizer]);

  const virtualRows = virtualizer.getVirtualItems();

  return (
    <div class={`filmstrip${hidden ? " filmstrip--hidden" : ""}`}>
      <div class="filmstrip__scroll" ref={scrollRef}>
        <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
          {virtualRows.map((virtual) => {
            const row = rows[virtual.index];
            if (!row) return null;
            return (
              <div
                key={virtual.key}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  transform: `translateY(${virtual.start}px)`,
                }}
              >
                {row.kind === "header" ? (
                  <div class={`filmstrip__group${row.warn ? " filmstrip__group--warn" : ""}`}>
                    <span>{row.warn ? `⚠ ${row.label}` : row.label}</span>
                    <span class="filmstrip__count">{row.count}</span>
                  </div>
                ) : (
                  <StripRow
                    dataset={dataset}
                    item={row.item!}
                    selected={row.item!.stem === selected}
                    onSelect={onSelect}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StripRow({
  dataset,
  item,
  selected,
  onSelect,
}: {
  dataset: string;
  item: Item;
  selected: boolean;
  onSelect(stem: string): void;
}) {
  const empty = item.boxes === 0;
  return (
    <button
      class={`strip-row${empty ? " strip-row--empty" : ""}`}
      aria-selected={selected}
      onClick={() => onSelect(item.stem)}
      title={item.filename}
    >
      <img
        class="strip-row__thumb"
        src={assets.thumb(dataset, item.stem, 160, item.token)}
        alt=""
        loading="lazy"
        decoding="async"
        width={44}
        height={44}
      />
      <span class="strip-row__body">
        <span class="strip-row__stem">{item.stem}</span>
        <span class={`strip-row__meta${empty ? " strip-row__meta--warn" : ""}`}>
          <span
            class={`pill${item.reviewed ? " pill--reviewed" : empty ? " pill--warn" : ""}`}
            aria-hidden="true"
          />
          {empty ? "0 boxes" : `${item.boxes} ${item.boxes === 1 ? "box" : "boxes"}`}
          {item.reviewed ? " · reviewed" : ""}
        </span>
      </span>
    </button>
  );
}
