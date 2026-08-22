/**
 * Box geometry, and the undo stack that edits go through.
 *
 * Boxes are *source geometry* (doc 08). The mask is derived from them by
 * expansion and feather and re-derives live; nothing here knows about the
 * mask, and nothing about the mask writes back here.
 *
 * Every mutation is a command with do/undo, scoped to one image. Doc 10
 * says retrofitting undo is painful and it goes in from the start — so it
 * is the only way boxes change.
 */

import type { Box, BoxShape } from "~/api/types";

export const MANUAL = "manual";

/**
 * The default shape. A box file written before shapes existed holds
 * rectangles, and the daemon omits the field for them — so `undefined`
 * means `rect` everywhere, and reading it goes through here rather than
 * being re-derived at each call site.
 */
export const RECT: BoxShape = "rect";
export const ELLIPSE: BoxShape = "ellipse";

export function shapeOf(box: { shape?: BoxShape }): BoxShape {
  return box.shape === ELLIPSE ? ELLIPSE : RECT;
}

export function isEllipse(box: { shape?: BoxShape }): boolean {
  return shapeOf(box) === ELLIPSE;
}

/** Doc 10's minimum. Deep enough that a review pass cannot outrun it. */
export const UNDO_DEPTH = 100;

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The eight resize handles, plus the body for moving. */
export type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

export function normalise(rect: Rect): Rect {
  // A drag up-and-left produces negative width; store it the other way up.
  const x = rect.w < 0 ? rect.x + rect.w : rect.x;
  const y = rect.h < 0 ? rect.y + rect.h : rect.y;
  return { x, y, w: Math.abs(rect.w), h: Math.abs(rect.h) };
}

/** Clamp a box inside the image. Doc 08: a box cannot be dragged off-canvas. */
export function clamp(rect: Rect, width: number, height: number): Rect {
  const w = Math.min(Math.round(rect.w), width);
  const h = Math.min(Math.round(rect.h), height);
  return {
    x: Math.max(0, Math.min(Math.round(rect.x), width - w)),
    y: Math.max(0, Math.min(Math.round(rect.y), height - h)),
    w,
    h,
  };
}

/**
 * Is this point inside the region — the *shape*, not its bounding box.
 *
 * Ellipses are tested against the ellipse. Two of them overlapping at the
 * corners is the common case in a group shot, and hit-testing bounding
 * boxes there selects the face you are not pointing at.
 */
export function contains(box: Box, point: { x: number; y: number }): boolean {
  const inBounds =
    point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h;
  if (!inBounds || !isEllipse(box)) return inBounds;

  const rx = box.w / 2;
  const ry = box.h / 2;
  if (rx <= 0 || ry <= 0) return false;
  const dx = (point.x - (box.x + rx)) / rx;
  const dy = (point.y - (box.y + ry)) / ry;
  return dx * dx + dy * dy <= 1;
}

export function intersects(a: Rect, b: Box): boolean {
  return !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
}

export function area(box: Box): number {
  return Math.max(0, box.w) * Math.max(0, box.h);
}

/** Apply a handle drag to a rect, in image coordinates. */
export function resize(box: Box, handle: Handle, dx: number, dy: number): Rect {
  let { x, y, w, h } = box;
  if (handle.includes("n")) {
    y += dy;
    h -= dy;
  }
  if (handle.includes("s")) h += dy;
  if (handle.includes("w")) {
    x += dx;
    w -= dx;
  }
  if (handle.includes("e")) w += dx;
  return normalise({ x, y, w, h });
}

/** The centre of a handle, in image coordinates. */
export function handlePoint(box: Box, handle: Handle): { x: number; y: number } {
  const midX = box.x + box.w / 2;
  const midY = box.y + box.h / 2;
  const right = box.x + box.w;
  const bottom = box.y + box.h;
  switch (handle) {
    case "nw":
      return { x: box.x, y: box.y };
    case "n":
      return { x: midX, y: box.y };
    case "ne":
      return { x: right, y: box.y };
    case "e":
      return { x: right, y: midY };
    case "se":
      return { x: right, y: bottom };
    case "s":
      return { x: midX, y: bottom };
    case "sw":
      return { x: box.x, y: bottom };
    case "w":
      return { x: box.x, y: midY };
  }
}

export const HANDLE_CURSOR: Record<Handle, string> = {
  nw: "nwse-resize",
  n: "ns-resize",
  ne: "nesw-resize",
  e: "ew-resize",
  se: "nwse-resize",
  s: "ns-resize",
  sw: "nesw-resize",
  w: "ew-resize",
};

// --------------------------------------------------------------------------
// expansion — the same arithmetic the server uses
// --------------------------------------------------------------------------

/**
 * Grow a box about its centre, biased upward.
 *
 * This must match `Box.expanded` in `core/detect/base.py` exactly. The
 * client is a preview of a file the server writes, and a preview that
 * disagrees with the artifact is worse than no preview: it would show a
 * face covered that is not.
 */
export function expand(box: Box, factor: number, upBias: number): Rect {
  const growX = (box.w * factor - box.w) / 2;
  const growY = (box.h * factor - box.h) / 2;
  const top = growY * upBias;
  return {
    x: Math.round(box.x - growX),
    y: Math.round(box.y - top),
    w: Math.round(box.w + growX * 2),
    h: Math.round(box.h + top + growY),
  };
}

// --------------------------------------------------------------------------
// undo stack
// --------------------------------------------------------------------------

/** One reversible edit: the box list before and after. */
interface Edit {
  label: string;
  before: Box[];
  after: Box[];
}

/**
 * Box edits for one image, with undo/redo.
 *
 * Snapshots rather than fine-grained commands: a box list is a handful of
 * small objects, so storing both sides of an edit costs nothing and
 * removes every chance of an inverse being subtly wrong.
 */
export class BoxHistory {
  private undoStack: Edit[] = [];
  private redoStack: Edit[] = [];

  constructor(private current: Box[] = []) {}

  get boxes(): Box[] {
    return this.current;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  /** True once anything has been changed since the last `reset`. */
  get dirty(): boolean {
    return this.undoStack.length > 0;
  }

  /** Replace the contents without recording an edit. For loading an image. */
  reset(boxes: Box[]): void {
    this.current = boxes;
    this.undoStack = [];
    this.redoStack = [];
  }

  /** Record a change. Returns the new box list. */
  apply(label: string, next: Box[]): Box[] {
    this.undoStack.push({ label, before: this.current, after: next });
    if (this.undoStack.length > UNDO_DEPTH) this.undoStack.shift();
    this.redoStack = [];
    this.current = next;
    return next;
  }

  undo(): Box[] {
    const edit = this.undoStack.pop();
    if (!edit) return this.current;
    this.redoStack.push(edit);
    this.current = edit.before;
    return this.current;
  }

  redo(): Box[] {
    const edit = this.redoStack.pop();
    if (!edit) return this.current;
    this.undoStack.push(edit);
    this.current = edit.after;
    return this.current;
  }
}
