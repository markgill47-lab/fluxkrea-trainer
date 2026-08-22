/**
 * The dataset gallery — doc 09.
 *
 * "Filter chips below it are the primary navigation of a large dataset and
 * carry live counts. Selection is the input to every batch operation, so
 * the selection count is always visible in the header."
 *
 * Both of those shape this file. The filters are derived from the item
 * list rather than fetched, so their counts cannot disagree with what the
 * grid is showing; and every batch action reads the same selection.
 *
 * **Detect faces and Export masks are not here.** They live on the Masks
 * screen, where their result is actually looked at. Pressing a button on
 * one tab and having to switch to another to see whether it did the right
 * thing is how a detect pass gets run twice.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Item, Task, ValidationReport } from "~/api/types";
import { GalleryInspectorHost } from "./InspectorHost";
import { ResizeDialog } from "./ResizeDialog";
import { ThumbnailGrid } from "./ThumbnailGrid";
import { ValidationPanel } from "./ValidationPanel";

interface Props {
  dataset: string;
  onError(message: string | null): void;
  onOpenReview(stem: string): void;
}

type FilterId = "all" | "unmasked" | "no-caption" | "unreviewed" | "no-boxes" | "rated-bad";

interface Filter {
  id: FilterId;
  label: string;
  match(item: Item): boolean;
  warn?: boolean;
}

/** The filters that correspond to something actually being wrong. */
const FILTERS: Filter[] = [
  { id: "all", label: "all", match: () => true },
  { id: "unmasked", label: "unmasked", match: (item) => !item.has_mask, warn: true },
  { id: "no-caption", label: "no caption", match: (item) => !item.has_caption, warn: true },
  { id: "no-boxes", label: "no boxes", match: (item) => item.boxes === 0, warn: true },
  { id: "unreviewed", label: "unreviewed", match: (item) => !item.reviewed },
  { id: "rated-bad", label: "rated bad", match: (item) => item.quality === "bad" },
];

/** Cell sizes cycled by `[` and `]` (doc 09). */
const CELL_SIZES = [96, 128, 160, 200, 256, 320];

export function GalleryScreen({ dataset, onError, onOpenReview }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [filter, setFilter] = useState<FilterId>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [focused, setFocused] = useState<string | null>(null);
  const [cellIndex, setCellIndex] = useState(2);
  const [task, setTask] = useState<Task | null>(null);
  const [loading, setLoading] = useState(true);
  const [showProblems, setShowProblems] = useState(false);
  const [resizing, setResizing] = useState(false);
  const anchor = useRef<string | null>(null);

  const cell = CELL_SIZES[cellIndex] ?? 160;

  // -- data ----------------------------------------------------------------

  const reload = useCallback(async () => {
    try {
      const [payload, validation] = await Promise.all([
        api.items(dataset),
        api.validate(dataset).catch(() => null),
      ]);
      setItems(payload.items);
      setReport(validation);
      onError(null);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [dataset, onError]);

  useEffect(() => {
    setLoading(true);
    setSelected(new Set());
    setFocused(null);
    void reload();
  }, [reload]);

  // Counts come from the same array the grid renders, so a chip can never
  // claim a number the grid does not show.
  const counts = useMemo(() => {
    const out = {} as Record<FilterId, number>;
    for (const entry of FILTERS) out[entry.id] = items.filter(entry.match).length;
    return out;
  }, [items]);

  const visible = useMemo(() => {
    const active = FILTERS.find((entry) => entry.id === filter) ?? FILTERS[0]!;
    return items.filter(active.match);
  }, [items, filter]);

  const selectedItems = useMemo(
    () => items.filter((item) => selected.has(item.stem)),
    [items, selected],
  );

  const focusedItem = focused ? items.find((item) => item.stem === focused) : undefined;

  // -- selection -----------------------------------------------------------

  const select = useCallback(
    (stem: string, event: MouseEvent) => {
      setFocused(stem);
      setSelected((current) => {
        const next = new Set(current);
        if (event.shiftKey && anchor.current) {
          // Range over what is *visible*, which is what the user sees.
          const from = visible.findIndex((item) => item.stem === anchor.current);
          const to = visible.findIndex((item) => item.stem === stem);
          if (from >= 0 && to >= 0) {
            for (const item of visible.slice(Math.min(from, to), Math.max(from, to) + 1)) {
              next.add(item.stem);
            }
            return next;
          }
        }
        if (event.ctrlKey || event.metaKey) {
          next.has(stem) ? next.delete(stem) : next.add(stem);
          anchor.current = stem;
          return next;
        }
        anchor.current = stem;
        return new Set([stem]);
      });
    },
    [visible],
  );

  // -- operations ----------------------------------------------------------

  /** Submit an operation and follow it to completion over SSE. */
  const runOperation = useCallback(
    async (operation: string, options: Record<string, unknown> = {}) => {
      try {
        const started = await api.runOperation(dataset, operation, options);
        setTask(started);

        let status = started.status;
        while (status === "queued" || status === "running") {
          await new Promise((resolve) => setTimeout(resolve, 350));
          const latest = await api.task(started.id);
          setTask(latest);
          status = latest.status;
        }
        await reload();
        // Leave the terminal state on screen briefly so a fast operation
        // does not just flicker.
        setTimeout(() => setTask(null), 2500);
      } catch (error) {
        if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
        setTask(null);
      }
    },
    [dataset, reload, onError],
  );

  const setCaption = useCallback(
    async (stem: string, caption: string) => {
      await api.putCaption(dataset, stem, caption);
      setItems((current) =>
        current.map((item) =>
          item.stem === stem ? { ...item, caption, has_caption: caption.trim().length > 0 } : item,
        ),
      );
    },
    [dataset],
  );

  const setQuality = useCallback(
    async (stems: string[], quality: string | null) => {
      try {
        await Promise.all(stems.map((stem) => api.putQuality(dataset, stem, quality)));
        setItems((current) =>
          current.map((item) => (stems.includes(item.stem) ? { ...item, quality } : item)),
        );
      } catch (error) {
        if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
      }
    },
    [dataset, onError],
  );

  const appendToCaptions = useCallback(
    async (text: string) => {
      const targets = selectedItems;
      await Promise.all(
        targets.map((item) => {
          const existing = (item.caption ?? "").trim();
          const next = existing ? `${existing}, ${text}` : text;
          return setCaption(item.stem, next);
        }),
      );
    },
    [selectedItems, setCaption],
  );

  // -- keyboard ------------------------------------------------------------

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      if (event.key === "[") setCellIndex((index) => Math.max(0, index - 1));
      if (event.key === "]") setCellIndex((index) => Math.min(CELL_SIZES.length - 1, index + 1));

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
        event.preventDefault();
        setSelected(new Set(visible.map((item) => item.stem)));
      }
      if (event.key === "Escape") setSelected(new Set());
      if (event.key === "Enter" && focused) onOpenReview(focused);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, focused, onOpenReview]);

  const scope = selected.size > 0 ? `${selected.size} selected` : `${visible.length} shown`;
  const problems = report ? report.problems.length : 0;

  return (
    <div class="gallery">
      <header class="gallery__head">
        <span class="gallery__title">{dataset}</span>
        <span class="gallery__meta tabular">{items.length} items</span>
        {problems > 0 && (
          <button
            class={`chip${report && !report.ok ? " chip--warn" : ""}`}
            onClick={() => setShowProblems((open) => !open)}
            aria-pressed={showProblems}
          >
            {report && !report.ok ? "⚠" : "•"} {problems} {problems === 1 ? "issue" : "issues"}
          </button>
        )}
        <span class="topbar__spacer" />
        <span class="gallery__scope tabular">{scope}</span>
        <button class="btn" onClick={() => void runOperation("validate")} disabled={!!task}>
          Validate
        </button>
        <button
          class="btn"
          onClick={() => setResizing(true)}
          disabled={!!task}
          title="Fit every image's longest edge to 1024 or 2048"
        >
          Resize
        </button>
        <button
          class="btn"
          onClick={() => void runOperation("caption", {})}
          disabled={!!task}
          title="Caption images that have none, with the configured backend"
        >
          Caption
        </button>
      </header>

      <div class="gallery__filters">
        {FILTERS.map((entry) => {
          const count = counts[entry.id] ?? 0;
          if (entry.id !== "all" && count === 0) return null;
          return (
            <button
              key={entry.id}
              class={`chip${filter === entry.id ? " chip--on" : ""}${
                entry.warn && count > 0 && filter !== entry.id ? " chip--warn" : ""
              }`}
              aria-pressed={filter === entry.id}
              onClick={() => setFilter(entry.id)}
            >
              {entry.label}
              <span class="chip__count">{count}</span>
            </button>
          );
        })}
        <span class="topbar__spacer" />
        <span class="hint">
          cell <kbd>[</kbd> <kbd>]</kbd> · {cell}px
        </span>
      </div>

      {task && (
        <div class={`taskbar${task.status === "failed" ? " taskbar--failed" : ""}`}>
          <span>{task.detail.operation ?? task.kind}</span>
          {task.detail.progress && task.detail.progress.total > 0 && (
            <>
              <progress value={task.detail.progress.step} max={task.detail.progress.total} />
              <span class="tabular">
                {task.detail.progress.step}/{task.detail.progress.total}
              </span>
            </>
          )}
          <span class="taskbar__status">{task.error || task.status}</span>
        </div>
      )}

      {showProblems && report && (
        <ValidationPanel
          report={report}
          onPick={(stem) => {
            setFocused(stem);
            setSelected(new Set([stem]));
            setFilter("all");
          }}
          onClose={() => setShowProblems(false)}
        />
      )}

      {loading ? (
        <div class="empty">
          <span class="spinner" />
        </div>
      ) : (
        <ThumbnailGrid
          dataset={dataset}
          items={visible}
          selected={selected}
          focused={focused}
          cell={cell}
          onSelect={select}
          onOpen={onOpenReview}
        />
      )}

      <GalleryInspectorHost
        dataset={dataset}
        selectedItems={selectedItems}
        focusedItem={focusedItem}
        onCaption={setCaption}
        onQuality={(stem, quality) => void setQuality([stem], quality)}
        onQualityAll={(quality) => void setQuality([...selected], quality)}
        onAppend={appendToCaptions}
        onClear={() => setSelected(new Set())}
        onOpen={onOpenReview}
      />

      {resizing && (
        <ResizeDialog
          items={items}
          running={!!task}
          onClose={() => setResizing(false)}
          onRun={(size, upscale) => {
            setResizing(false);
            void runOperation("resize", { size, upscale });
          }}
        />
      )}
    </div>
  );
}
