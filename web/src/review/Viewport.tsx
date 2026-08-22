/**
 * The image viewport: three layers sharing one transform (doc 10).
 *
 *   <canvas>  image + derived mask compositing   raster
 *   <svg>     boxes, handles, marquee            vector
 *   <div>     readouts                           DOM
 *
 * The mask is derived live from box geometry, not fetched — expansion and
 * feather change interactively and a round-trip per adjustment is not
 * viable. The *exported* mask is produced server-side from the same
 * geometry, so the preview and the trained artifact cannot diverge; this
 * canvas's job is to be an accurate preview, not the source of the file.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Box, BoxShape } from "~/api/types";
import {
  clamp,
  contains,
  ELLIPSE,
  expand,
  HANDLES,
  HANDLE_CURSOR,
  handlePoint,
  intersects,
  isEllipse,
  MANUAL,
  normalise,
  resize,
  type Handle,
  type Rect,
} from "~/lib/boxes";
import * as vp from "~/lib/viewport";

export type MaskMode = "off" | "overlay" | "isolate";

/**
 * One region's outline. Rect or ellipse, decided here and nowhere else.
 *
 * Every place that draws a region — the outline, its shadow, the live
 * drag — goes through this, so there is no way for one of them to keep
 * drawing rectangles after the others learned about ellipses.
 */
function Outline({ box, className }: { box: Box; className: string }) {
  if (isEllipse(box)) {
    return (
      <ellipse
        class={className}
        cx={box.x + box.w / 2}
        cy={box.y + box.h / 2}
        rx={box.w / 2}
        ry={box.h / 2}
      />
    );
  }
  return <rect class={className} x={box.x} y={box.y} width={box.w} height={box.h} />;
}

export interface MaskSettings {
  expand: number;
  expandUp: number;
  feather: number;
  opacity: number;
}

interface Props {
  bitmap: ImageBitmap | null;
  boxes: Box[];
  selected: number[];
  maskMode: MaskMode;
  showDetected: boolean;
  mask: MaskSettings;
  drawMode: boolean;
  /** What a new hand-drawn region is. Detection produces ellipses; this is
   *  the shape the pointer draws, which is a separate choice. */
  drawShape: BoxShape;
  onSelect(indices: number[]): void;
  onChange(label: string, boxes: Box[]): void;
  onTransform(transform: vp.Transform): void;
  transformRef: { current: vp.Transform };
}

/** Handle size in *screen* pixels, so it is constant at any zoom (doc 10). */
const HANDLE_SIZE = 8;

/** How close, in screen pixels, a pointer must be to grab a handle. */
const HANDLE_SLOP = 10;

type Drag =
  | { kind: "none" }
  | { kind: "pan"; lastX: number; lastY: number }
  | { kind: "draw"; origin: { x: number; y: number }; rect: Rect }
  | { kind: "move"; start: { x: number; y: number }; original: Box[] }
  | { kind: "resize"; handle: Handle; start: { x: number; y: number }; original: Box }
  | { kind: "marquee"; origin: { x: number; y: number }; rect: Rect };

export function Viewport({
  bitmap,
  boxes,
  selected,
  maskMode,
  showDetected,
  mask,
  drawMode,
  drawShape,
  onSelect,
  onChange,
  onTransform,
  transformRef,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<Drag>({ kind: "none" });
  const spaceRef = useRef(false);

  const [size, setSize] = useState<vp.Size>({ width: 0, height: 0 });
  const [transform, setTransform] = useState<vp.Transform>(vp.identity());
  const [live, setLive] = useState<Rect | null>(null);
  const [cursor, setCursor] = useState<string>("default");

  transformRef.current = transform;

  const image: vp.Size = bitmap
    ? { width: bitmap.width, height: bitmap.height }
    : { width: 0, height: 0 };

  // -- sizing --------------------------------------------------------------

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  // Fit whenever a new image arrives. Stepping through a review pass should
  // land every image at a comparable size without touching anything.
  useEffect(() => {
    if (!bitmap || !size.width) return;
    const fitted = vp.fit({ width: bitmap.width, height: bitmap.height }, size);
    setTransform(fitted);
    onTransform(fitted);
  }, [bitmap, size.width, size.height]);

  const update = useCallback(
    (next: vp.Transform) => {
      const snapped = vp.snapped(next);
      setTransform(snapped);
      onTransform(snapped);
    },
    [onTransform],
  );

  // -- mask compositing ----------------------------------------------------

  /**
   * Render the derived mask to an offscreen canvas at image resolution.
   *
   * Kept off the main draw so a slider drag does not re-decode the image;
   * doc 10 wants 60fps on a 4K image while dragging feather.
   */
  const renderMask = useCallback(() => {
    if (!bitmap) return null;
    let canvas = maskCanvasRef.current;
    if (!canvas || canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      maskCanvasRef.current = canvas;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fff";
    // Feather is a deliberate gradient at the boundary in a mask that is
    // otherwise hard-edged (doc 03) — never acquired through a resize.
    ctx.filter = mask.feather > 0 ? `blur(${mask.feather / 2}px)` : "none";
    for (const box of boxes) {
      const grown = expand(box, mask.expand, mask.expandUp);
      if (isEllipse(box)) {
        // Not clamped, deliberately — the same rule as `render_mask` in
        // `core/dataset/ops/mask.py`. Clamping an ellipse clamps its
        // bounding box, which moves the centre and squashes the axes, so
        // the preview would show a different ellipse from the one exported
        // for any face near the edge of the frame. The canvas clips.
        if (grown.w > 0 && grown.h > 0) {
          ctx.beginPath();
          ctx.ellipse(
            grown.x + grown.w / 2,
            grown.y + grown.h / 2,
            grown.w / 2,
            grown.h / 2,
            0,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
        continue;
      }
      const clipped = clamp(grown, bitmap.width, bitmap.height);
      if (clipped.w > 0 && clipped.h > 0) ctx.fillRect(clipped.x, clipped.y, clipped.w, clipped.h);
    }
    ctx.filter = "none";
    return canvas;
  }, [bitmap, boxes, mask.expand, mask.expandUp, mask.feather]);

  // -- drawing -------------------------------------------------------------

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size.width) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    if (!bitmap) return;

    const { scale, tx, ty } = transform;
    // Nearest-neighbour at or above 100% so a mask edge is shown as it is,
    // not as bilinear smear (doc 10).
    ctx.imageSmoothingEnabled = scale < 1;
    ctx.imageSmoothingQuality = "high";

    const width = bitmap.width * scale;
    const height = bitmap.height * scale;

    if (maskMode !== "isolate") {
      ctx.drawImage(bitmap, tx, ty, width, height);
    } else {
      // Isolate: verify coverage without the photograph distracting.
      ctx.fillStyle = "#000";
      ctx.fillRect(tx, ty, width, height);
    }

    if (maskMode !== "off") {
      const maskCanvas = renderMask();
      if (maskCanvas) {
        ctx.save();
        ctx.globalAlpha = maskMode === "isolate" ? 1 : mask.opacity;
        // The magenta fill, tinted through the mask's alpha.
        ctx.globalCompositeOperation = "source-over";
        const tint = document.createElement("canvas");
        tint.width = maskCanvas.width;
        tint.height = maskCanvas.height;
        const tintCtx = tint.getContext("2d");
        if (tintCtx) {
          tintCtx.drawImage(maskCanvas, 0, 0);
          tintCtx.globalCompositeOperation = "source-in";
          tintCtx.fillStyle =
            getComputedStyle(document.documentElement).getPropertyValue("--overlay-mask").trim() ||
            "#FF00AA";
          tintCtx.fillRect(0, 0, tint.width, tint.height);
          ctx.drawImage(tint, tx, ty, width, height);
        }
        ctx.restore();
      }
    }
  }, [bitmap, transform, size, maskMode, mask.opacity, renderMask]);

  // -- pointer interaction -------------------------------------------------

  const local = (event: PointerEvent): { x: number; y: number } => {
    const rect = hostRef.current!.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const hitHandle = (point: { x: number; y: number }): Handle | null => {
    if (selected.length !== 1) return null;
    const box = boxes[selected[0]!];
    if (!box) return null;
    const slop = vp.screenToImageLength(transform, HANDLE_SLOP);
    for (const handle of HANDLES) {
      const at = handlePoint(box, handle);
      if (Math.abs(point.x - at.x) <= slop && Math.abs(point.y - at.y) <= slop) return handle;
    }
    return null;
  };

  const hitBox = (point: { x: number; y: number }): number => {
    // Topmost first, so a small box drawn over a large one is reachable.
    for (let index = boxes.length - 1; index >= 0; index -= 1) {
      const box = boxes[index]!;
      if (!showDetected && box.src !== MANUAL) continue;
      if (contains(box, point)) return index;
    }
    return -1;
  };

  const onPointerDown = (event: PointerEvent) => {
    if (!bitmap || event.button === 2) return;
    const host = hostRef.current!;
    host.setPointerCapture(event.pointerId);
    const point = vp.toImage(transform, local(event));

    // Middle-drag or held space pans, whatever else is going on.
    if (event.button === 1 || spaceRef.current) {
      dragRef.current = { kind: "pan", lastX: event.clientX, lastY: event.clientY };
      setCursor("grabbing");
      return;
    }

    const handle = hitHandle(point);
    if (handle) {
      dragRef.current = {
        kind: "resize",
        handle,
        start: point,
        original: boxes[selected[0]!]!,
      };
      return;
    }

    const index = hitBox(point);

    if (drawMode || (index === -1 && event.shiftKey === false && selected.length === 0)) {
      if (drawMode) {
        dragRef.current = { kind: "draw", origin: point, rect: { ...point, w: 0, h: 0 } };
        setLive({ ...point, w: 0, h: 0 });
        return;
      }
    }

    if (index === -1) {
      dragRef.current = { kind: "marquee", origin: point, rect: { ...point, w: 0, h: 0 } };
      setLive({ ...point, w: 0, h: 0 });
      if (!event.shiftKey) onSelect([]);
      return;
    }

    const next = event.shiftKey
      ? selected.includes(index)
        ? selected.filter((i) => i !== index)
        : [...selected, index]
      : selected.includes(index)
        ? selected
        : [index];
    onSelect(next);
    dragRef.current = {
      kind: "move",
      start: point,
      original: next.map((i) => boxes[i]!),
    };
  };

  const onPointerMove = (event: PointerEvent) => {
    const drag = dragRef.current;
    const point = bitmap ? vp.toImage(transform, local(event)) : { x: 0, y: 0 };

    if (drag.kind === "none") {
      // Hover feedback: handles take priority over the body.
      if (!bitmap) return;
      const handle = hitHandle(point);
      setCursor(handle ? HANDLE_CURSOR[handle] : hitBox(point) >= 0 ? "move" : "default");
      return;
    }

    if (drag.kind === "pan") {
      update(vp.pan(transform, event.clientX - drag.lastX, event.clientY - drag.lastY));
      dragRef.current = { kind: "pan", lastX: event.clientX, lastY: event.clientY };
      return;
    }

    if (!bitmap) return;

    if (drag.kind === "draw" || drag.kind === "marquee") {
      const rect = normalise({
        x: drag.origin.x,
        y: drag.origin.y,
        w: point.x - drag.origin.x,
        h: point.y - drag.origin.y,
      });
      dragRef.current = { ...drag, rect };
      setLive(rect);
      return;
    }

    if (drag.kind === "move") {
      const dx = point.x - drag.start.x;
      const dy = point.y - drag.start.y;
      const next = boxes.slice();
      selected.forEach((index, position) => {
        const original = drag.original[position];
        if (!original) return;
        const moved = clamp(
          { x: original.x + dx, y: original.y + dy, w: original.w, h: original.h },
          bitmap.width,
          bitmap.height,
        );
        next[index] = { ...original, ...moved };
      });
      onChange("move", next);
      return;
    }

    if (drag.kind === "resize") {
      const rect = clamp(
        resize(drag.original, drag.handle, point.x - drag.start.x, point.y - drag.start.y),
        bitmap.width,
        bitmap.height,
      );
      const next = boxes.slice();
      next[selected[0]!] = { ...drag.original, ...rect };
      onChange("resize", next);
    }
  };

  const onPointerUp = (event: PointerEvent) => {
    const drag = dragRef.current;
    dragRef.current = { kind: "none" };
    setLive(null);
    setCursor("default");
    hostRef.current?.releasePointerCapture(event.pointerId);
    if (!bitmap) return;

    if (drag.kind === "draw") {
      const rect = clamp(drag.rect, bitmap.width, bitmap.height);
      // Ignore an accidental click that produced a degenerate box.
      if (rect.w < 4 || rect.h < 4) return;
      const next = [...boxes, { ...rect, src: MANUAL, conf: null, shape: drawShape }];
      onChange("draw box", next);
      onSelect([next.length - 1]);
      return;
    }

    if (drag.kind === "marquee") {
      const rect = drag.rect;
      if (rect.w < 3 && rect.h < 3) return;
      const hits = boxes
        .map((box, index) => ({ box, index }))
        .filter(({ box }) => (showDetected || box.src === MANUAL) && intersects(rect, box))
        .map(({ index }) => index);
      onSelect(hits);
    }
  };

  // -- wheel zoom ----------------------------------------------------------

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = host.getBoundingClientRect();
      const pivot = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      // Trackpads report small deltas continuously; a fixed step per event
      // would make them unusable, so scale by the delta itself.
      const factor = Math.exp(-event.deltaY * 0.0015);
      update(vp.zoomAt(transformRef.current, pivot, factor));
    };
    host.addEventListener("wheel", onWheel, { passive: false });
    return () => host.removeEventListener("wheel", onWheel);
  }, [update]);

  // Space held pans; tapped it marks reviewed (handled by the screen).
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (event.code === "Space") spaceRef.current = true;
    };
    const up = (event: KeyboardEvent) => {
      if (event.code === "Space") spaceRef.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const handleScale = vp.screenToImageLength(transform, HANDLE_SIZE);
  const visible = showDetected ? boxes : boxes.filter((box) => box.src === MANUAL);

  return (
    <div
      ref={hostRef}
      class={`viewport${drawMode ? " viewport--drawing" : ""}`}
      style={{ cursor: drawMode ? "crosshair" : cursor }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <canvas ref={canvasRef} class="viewport__layer viewport__canvas" />

      {bitmap && size.width > 0 && (
        <svg
          class="viewport__layer viewport__svg"
          viewBox={vp.viewBox(transform, size)}
          preserveAspectRatio="none"
        >
          {visible.map((box) => {
            const index = boxes.indexOf(box);
            const isSelected = selected.includes(index);
            const kind = box.src === MANUAL ? "manual" : "detected";
            return (
              <g key={`${box.x}-${box.y}-${box.w}-${box.h}-${index}`}>
                <Outline box={box} className="box__shadow" />
                <Outline box={box} className={`box box--${isSelected ? "selected" : kind}`} />
              </g>
            );
          })}

          {selected.length === 1 &&
            boxes[selected[0]!] &&
            HANDLES.map((handle) => {
              const at = handlePoint(boxes[selected[0]!]!, handle);
              return (
                <rect
                  key={handle}
                  class="box__handle"
                  x={at.x - handleScale / 2}
                  y={at.y - handleScale / 2}
                  width={handleScale}
                  height={handleScale}
                />
              );
            })}

          {live &&
            (dragRef.current.kind === "marquee" ? (
              // The marquee is a selection rectangle whatever is being
              // drawn — it is a region of the screen, not a region of the
              // image, and drawing it as an ellipse would misdescribe
              // which boxes it is about to catch.
              <rect class="marquee" x={live.x} y={live.y} width={live.w} height={live.h} />
            ) : (
              <Outline
                box={{ ...live, src: MANUAL, shape: drawShape }}
                className="box box--manual"
              />
            ))}
        </svg>
      )}

      {!bitmap && (
        <div class="empty">
          <div class="empty__title">No image</div>
          <div>Select an item from the filmstrip</div>
        </div>
      )}

      {drawMode && bitmap && (
        <div class="viewport__hint">
          Drag to draw {drawShape === ELLIPSE ? "an ellipse" : "a box"} · Esc to cancel
        </div>
      )}
    </div>
  );
}
