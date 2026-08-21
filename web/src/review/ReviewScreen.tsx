/**
 * Mask review — doc 09's keyboard map, in full.
 *
 * "The review pass must be completable without the mouse except for
 * drawing." That is the design constraint that shapes this file: every
 * action has a key, navigation saves and advances, and the only thing a
 * pointer is required for is drawing a box.
 *
 * On `Space` doing double duty (mark reviewed vs hold-to-pan), doc 09
 * flagged the ambiguity and the brief resolved it: Space marks reviewed
 * only, panning is middle-drag or held `H`. Both are implemented; the
 * hold-to-pan path in Viewport still honours a held Space for anyone with
 * the muscle memory, but a tap always marks.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api, ApiError, assets, isAbort } from "~/api/client";
import type { Box, Item, ReviewProgress } from "~/api/types";
import { BoxHistory, MANUAL } from "~/lib/boxes";
import * as vp from "~/lib/viewport";
import { Filmstrip } from "./Filmstrip";
import { Inspector } from "./Inspector";
import { Shortcuts } from "./Shortcuts";
import { Viewport } from "./Viewport";
import type { MaskMode, MaskSettings } from "./Viewport";

interface Props {
  dataset: string;
  detectors: string[];
  onError(message: string | null): void;
}

const DEFAULT_MASK: MaskSettings = {
  expand: 1.6,
  expandUp: 1.35,
  feather: 12,
  opacity: 0.35,
};

const MASK_CYCLE: MaskMode[] = ["off", "overlay", "isolate"];

export function ReviewScreen({ dataset, detectors, onError }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [progress, setProgress] = useState<ReviewProgress | null>(null);
  const [stem, setStem] = useState<string | null>(null);
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [maskMode, setMaskMode] = useState<MaskMode>("overlay");
  const [showDetected, setShowDetected] = useState(true);
  const [mask, setMask] = useState<MaskSettings>(DEFAULT_MASK);
  const [drawMode, setDrawMode] = useState(false);
  const [detector, setDetector] = useState(detectors[0] ?? "yunet");
  const [detecting, setDetecting] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [loading, setLoading] = useState(true);
  const [width, setWidth] = useState(1600);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const rootRef = useRef<HTMLDivElement>(null);
  const history = useRef(new BoxHistory());
  const transformRef = useRef<vp.Transform>(vp.identity());
  const reviewedRef = useRef(false);

  // Doc 09: below 1440px the inspector collapses to a toggle; below the
  // 1280px minimum the filmstrip goes too. Measured from the grid rather
  // than the window, because the rail comes out of the same budget.
  useEffect(() => {
    const host = rootRef.current;
    if (!host) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const narrow = width < 1440;
  const tiny = width < 1000;

  const ordered = useMemo(() => {
    // Same order the filmstrip renders, so J/K matches what is on screen.
    const empty = items.filter((item) => item.boxes === 0);
    const unreviewed = items.filter((item) => item.boxes > 0 && !item.reviewed);
    const reviewed = items.filter((item) => item.boxes > 0 && item.reviewed);
    return [...empty, ...unreviewed, ...reviewed];
  }, [items]);

  const position = stem ? ordered.findIndex((item) => item.stem === stem) : -1;
  const current = position >= 0 ? ordered[position] : undefined;

  // -- loading -------------------------------------------------------------

  const reload = useCallback(async () => {
    try {
      const payload = await api.items(dataset);
      setItems(payload.items);
      setProgress(payload.review);
      setStem((existing) => existing ?? payload.items[0]?.stem ?? null);
      onError(null);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [dataset, onError]);

  useEffect(() => {
    setLoading(true);
    void reload();
  }, [reload]);

  // Load the image and its boxes together. Both are cancelled when the
  // selection moves on, or a fast review pass queues work it never uses.
  useEffect(() => {
    if (!stem) return;
    const controller = new AbortController();
    let live = true;
    let decoded: ImageBitmap | null = null;

    (async () => {
      try {
        const [payload, response] = await Promise.all([
          api.boxes(dataset, stem, { signal: controller.signal }),
          fetch(assets.image(dataset, stem), { signal: controller.signal }),
        ]);
        if (!live) return;

        history.current.reset(payload.boxes);
        setBoxes(payload.boxes);
        setSelected([]);
        reviewedRef.current = payload.reviewed;

        // Decode off the main thread; the review loop must not stutter
        // while stepping through images (doc 10).
        decoded = await createImageBitmap(await response.blob());
        if (!live) {
          decoded.close();
          return;
        }
        setBitmap((previous) => {
          previous?.close();
          return decoded;
        });
      } catch (error) {
        if (!isAbort(error) && live) {
          onError(error instanceof ApiError ? error.message : String(error));
        }
      }
    })();

    return () => {
      live = false;
      controller.abort();
    };
  }, [dataset, stem, onError]);

  // -- editing -------------------------------------------------------------

  const change = useCallback((label: string, next: Box[]) => {
    setBoxes(history.current.apply(label, next));
  }, []);

  /**
   * Persist this image's boxes.
   *
   * Box edits are optimistic and local until here (doc 10) — one request
   * per image on navigation, not one per drag.
   */
  const save = useCallback(
    async (target: string, list: Box[], reviewed: boolean) => {
      try {
        await api.putBoxes(dataset, target, list, reviewed);
        setItems((existing) =>
          existing.map((item) =>
            item.stem === target ? { ...item, boxes: list.length, reviewed } : item,
          ),
        );
        const fresh = await api.review(dataset);
        setProgress(fresh);
      } catch (error) {
        if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
      }
    },
    [dataset, onError],
  );

  const go = useCallback(
    (delta: number) => {
      if (position < 0) return;
      const next = ordered[Math.min(ordered.length - 1, Math.max(0, position + delta))];
      if (!next || next.stem === stem) return;
      if (stem && history.current.dirty) void save(stem, boxes, reviewedRef.current);
      setStem(next.stem);
    },
    [ordered, position, stem, boxes, save],
  );

  const markReviewed = useCallback(
    (advance: boolean) => {
      if (!stem) return;
      reviewedRef.current = true;
      void save(stem, boxes, true);
      if (advance) go(1);
    },
    [stem, boxes, save, go],
  );

  const redetect = useCallback(async () => {
    if (!stem) return;
    setDetecting(true);
    try {
      // Whole-dataset detect with only_missing off would redo everything;
      // this is one image, so the operation runs and we re-read the boxes.
      const task = await api.runOperation(dataset, "detect", {
        detector,
        only_missing: false,
        workers: 1,
      });
      let status = task.status;
      while (status === "queued" || status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 400));
        status = (await api.task(task.id)).status;
      }
      const payload = await api.boxes(dataset, stem);
      history.current.reset(payload.boxes);
      setBoxes(payload.boxes);
      reviewedRef.current = payload.reviewed;
      await reload();
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setDetecting(false);
    }
  }, [dataset, stem, detector, reload, onError]);

  const deleteSelected = useCallback(() => {
    if (!selected.length) return;
    const drop = new Set(selected);
    change(
      "delete",
      boxes.filter((_, index) => !drop.has(index)),
    );
    setSelected([]);
  }, [selected, boxes, change]);

  const nudge = useCallback(
    (dx: number, dy: number) => {
      if (!selected.length || !bitmap) return;
      const drop = new Set(selected);
      change(
        "nudge",
        boxes.map((box, index) =>
          drop.has(index)
            ? {
                ...box,
                x: Math.max(0, Math.min(bitmap.width - box.w, box.x + dx)),
                y: Math.max(0, Math.min(bitmap.height - box.h, box.y + dy)),
              }
            : box,
        ),
      );
    },
    [selected, boxes, bitmap, change],
  );

  const setZoomTransform = useCallback((next: vp.Transform) => {
    transformRef.current = next;
    setZoom(vp.zoomPercent(next));
  }, []);

  // -- keyboard map --------------------------------------------------------

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      const step = event.shiftKey ? 10 : 1;
      const ctrl = event.ctrlKey || event.metaKey;

      if (ctrl && event.key.toLowerCase() === "z") {
        event.preventDefault();
        setBoxes(event.shiftKey ? history.current.redo() : history.current.undo());
        setSelected([]);
        return;
      }

      switch (event.key) {
        case "j":
        case "J":
        case "ArrowDown":
          if (!selected.length || event.key !== "ArrowDown") {
            event.preventDefault();
            go(1);
            return;
          }
          break;
        case "k":
        case "K":
        case "ArrowUp":
          if (!selected.length || event.key !== "ArrowUp") {
            event.preventDefault();
            go(-1);
            return;
          }
          break;
      }

      if (selected.length && event.key.startsWith("Arrow")) {
        event.preventDefault();
        if (event.key === "ArrowLeft") nudge(-step, 0);
        if (event.key === "ArrowRight") nudge(step, 0);
        if (event.key === "ArrowUp") nudge(0, -step);
        if (event.key === "ArrowDown") nudge(0, step);
        return;
      }

      switch (event.key) {
        case " ":
          event.preventDefault();
          markReviewed(!event.shiftKey);
          break;
        case "b":
        case "B":
          setDrawMode((on) => !on);
          break;
        case "Delete":
        case "Backspace":
          event.preventDefault();
          deleteSelected();
          break;
        case "Tab":
          if (boxes.length) {
            event.preventDefault();
            setSelected(([first]) => [((first ?? -1) + 1) % boxes.length]);
          }
          break;
        case "m":
        case "M":
          setMaskMode((mode) => MASK_CYCLE[(MASK_CYCLE.indexOf(mode) + 1) % MASK_CYCLE.length]!);
          break;
        case "d":
        case "D":
          setShowDetected((on) => !on);
          break;
        case "0":
          if (bitmap) {
            const host = document.querySelector(".viewport");
            if (host) {
              const rect = host.getBoundingClientRect();
              setZoomTransform(
                vp.fit(
                  { width: bitmap.width, height: bitmap.height },
                  { width: rect.width, height: rect.height },
                ),
              );
            }
          }
          break;
        case "1":
          if (bitmap) {
            const host = document.querySelector(".viewport");
            if (host) {
              const rect = host.getBoundingClientRect();
              setZoomTransform(
                vp.actualSize(
                  { width: bitmap.width, height: bitmap.height },
                  { width: rect.width, height: rect.height },
                ),
              );
            }
          }
          break;
        case "r":
        case "R":
          void redetect();
          break;
        case "?":
          setShowHelp((on) => !on);
          break;
        case "Escape":
          setDrawMode(false);
          setShowHelp(false);
          setSelected([]);
          break;
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, markReviewed, deleteSelected, nudge, boxes.length, bitmap, selected, redetect, setZoomTransform]);

  // Save whatever is pending when the screen goes away.
  useEffect(
    () => () => {
      if (stem && history.current.dirty) void save(stem, boxes, reviewedRef.current);
    },
    [stem, boxes, save],
  );

  const empties = progress?.empty.length ?? 0;

  // One expression for the grid, rather than a stack of modifier classes:
  // two independent collapses times a toggle is four combinations, and the
  // cascade is a poor place to keep track of them.
  const columns = [
    tiny ? "0" : narrow ? "200px" : "220px",
    "minmax(0, 1fr)",
    narrow ? "0" : "260px",
  ].join(" ");

  const layout = ["review", narrow ? "review--floating" : ""].filter(Boolean).join(" ");

  return (
    <div class={layout} ref={rootRef} style={{ gridTemplateColumns: columns }}>
      <header class="review__head">
        <span class="review__title">{dataset} · review</span>
        <span class="review__progress">
          {progress ? `${progress.reviewed}/${progress.total} reviewed` : loading ? "loading…" : ""}
        </span>
        {empties > 0 && (
          <span class="chip chip--warn">
            ⚠ {empties} with no detections — check these first
          </span>
        )}
        <span class="topbar__spacer" />
        {narrow && (
          <button
            class={`btn${inspectorOpen ? " btn--accent" : " btn--ghost"}`}
            onClick={() => setInspectorOpen((open) => !open)}
            aria-pressed={inspectorOpen}
          >
            Inspector
          </button>
        )}
        <button class="btn btn--ghost" onClick={() => setShowHelp(true)} title="Shortcuts (?)">
          ?
        </button>
      </header>

      <Filmstrip
        hidden={tiny}
        dataset={dataset}
        items={items}
        selected={stem}
        onSelect={(next) => {
          if (stem && history.current.dirty) void save(stem, boxes, reviewedRef.current);
          setStem(next);
        }}
      />

      <Viewport
        bitmap={bitmap}
        boxes={boxes}
        selected={selected}
        maskMode={maskMode}
        showDetected={showDetected}
        mask={mask}
        drawMode={drawMode}
        onSelect={setSelected}
        onChange={change}
        onTransform={setZoomTransform}
        transformRef={transformRef}
      />

      <div class="bottombar">
        <button class="btn btn--ghost" onClick={() => setMaskMode(nextMode(maskMode))}>
          Mask: {maskMode}
        </button>
        <button class="btn btn--ghost" onClick={() => setShowDetected((on) => !on)}>
          Detected: {showDetected ? "on" : "off"}
        </button>
        <button
          class={`btn${drawMode ? " btn--accent" : " btn--ghost"}`}
          onClick={() => setDrawMode((on) => !on)}
        >
          Draw box
        </button>
        <span class="bottombar__spacer" />
        <span class="bottombar__zoom tabular">{zoom}%</span>
        <button class="btn btn--ghost" onClick={() => go(-1)} disabled={position <= 0}>
          ◀
        </button>
        <span class="bottombar__position">
          {position >= 0 ? position + 1 : 0}/{ordered.length}
        </span>
        <button
          class="btn btn--ghost"
          onClick={() => go(1)}
          disabled={position < 0 || position >= ordered.length - 1}
        >
          ▶
        </button>
        <button class="btn btn--accent" onClick={() => markReviewed(true)} disabled={!current}>
          Mark reviewed
        </button>
      </div>

      {(!narrow || inspectorOpen) && (
      <Inspector
        boxes={boxes}
        selected={selected}
        mask={mask}
        detectors={detectors}
        detector={detector}
        detecting={detecting}
        onSelect={setSelected}
        onMask={(next) => setMask((existing) => ({ ...existing, ...next }))}
        onDetector={setDetector}
        onRedetect={() => void redetect()}
        onDelete={deleteSelected}
      />
      )}

      {showHelp && <Shortcuts onClose={() => setShowHelp(false)} />}
    </div>
  );
}

function nextMode(mode: MaskMode): MaskMode {
  return MASK_CYCLE[(MASK_CYCLE.indexOf(mode) + 1) % MASK_CYCLE.length]!;
}
