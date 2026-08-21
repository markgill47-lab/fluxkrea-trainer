/**
 * The pan/zoom transform. One object, one source of truth.
 *
 * Doc 10: the image viewport is three layers sharing one transform — a
 * canvas for the image and mask, an SVG for the boxes, and DOM for the
 * controls. A single `{scale, tx, ty}` drives all three, so there is never
 * a second opinion about where the image is.
 *
 * The rules here are the ones that make an image editor feel right or
 * broken, and none of them are negotiable:
 *
 * - **Zoom about the cursor**, not the centre.
 * - **Pixel snapping at ≥100%**, so a mask edge is shown as it is rather
 *   than as bilinear smear. A soft edge in the display that is not in the
 *   data sends someone chasing a bug that does not exist.
 * - **Handle sizes are screen-space**, so they stay 8px at any zoom.
 */

export interface Transform {
  /** Image pixels per screen pixel. 1 means 100%. */
  scale: number;
  /** Screen-space offset of the image origin, in CSS pixels. */
  tx: number;
  ty: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

/** Doc 10's clamp range. Below 5% nothing is legible; above 1600% is grain. */
export const MIN_SCALE = 0.05;
export const MAX_SCALE = 16;

/** Within this much of 100%, snap to it exactly. */
const SNAP_TOLERANCE = 0.02;

/** Padding around a fitted image, so it does not touch the panel edges. */
const FIT_PADDING = 24;

export function identity(): Transform {
  return { scale: 1, tx: 0, ty: 0 };
}

function clampScale(scale: number): number {
  const clamped = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
  return Math.abs(clamped - 1) < SNAP_TOLERANCE ? 1 : clamped;
}

/** The transform that centres *image* inside *viewport* at the largest fit. */
export function fit(image: Size, viewport: Size): Transform {
  if (!image.width || !image.height || !viewport.width || !viewport.height) {
    return identity();
  }
  const available = {
    width: Math.max(1, viewport.width - FIT_PADDING * 2),
    height: Math.max(1, viewport.height - FIT_PADDING * 2),
  };
  // Never scale up to fit: a 512px image in a 1400px panel should sit at
  // 100%, not be blown up to fill the space and look soft.
  const scale = Math.min(available.width / image.width, available.height / image.height, 1);
  return centred(image, viewport, clampScale(scale));
}

/** 100%, centred. */
export function actualSize(image: Size, viewport: Size): Transform {
  return centred(image, viewport, 1);
}

export function centred(image: Size, viewport: Size, scale: number): Transform {
  return {
    scale,
    tx: (viewport.width - image.width * scale) / 2,
    ty: (viewport.height - image.height * scale) / 2,
  };
}

/**
 * Zoom by *factor* keeping the image point under *pivot* stationary.
 *
 * The whole feel of a viewer lives in this function. Zooming about the
 * centre while the cursor is elsewhere makes the image squirm away from
 * where you are looking.
 */
export function zoomAt(current: Transform, pivot: Point, factor: number): Transform {
  const scale = clampScale(current.scale * factor);
  if (scale === current.scale) return current;

  // The image coordinate under the pivot must map back to the same screen
  // position after the scale change.
  const imageX = (pivot.x - current.tx) / current.scale;
  const imageY = (pivot.y - current.ty) / current.scale;

  return {
    scale,
    tx: pivot.x - imageX * scale,
    ty: pivot.y - imageY * scale,
  };
}

/** Zoom about the viewport centre. For keyboard `+` / `-`. */
export function zoomCentre(current: Transform, viewport: Size, factor: number): Transform {
  return zoomAt(current, { x: viewport.width / 2, y: viewport.height / 2 }, factor);
}

export function pan(current: Transform, dx: number, dy: number): Transform {
  return { scale: current.scale, tx: current.tx + dx, ty: current.ty + dy };
}

/**
 * Snap the translation to whole screen pixels when zoomed in.
 *
 * At or above 100% every image pixel covers at least one screen pixel, so
 * a fractional offset would resample a hard mask edge into a soft one.
 * Below 100% the image is being minified anyway and snapping would make
 * panning judder.
 */
export function snapped(current: Transform): Transform {
  if (current.scale < 1) return current;
  return { scale: current.scale, tx: Math.round(current.tx), ty: Math.round(current.ty) };
}

// --------------------------------------------------------------------------
// coordinate conversion
// --------------------------------------------------------------------------

/** Screen (viewport-relative CSS pixels) → image pixels. */
export function toImage(current: Transform, point: Point): Point {
  return {
    x: (point.x - current.tx) / current.scale,
    y: (point.y - current.ty) / current.scale,
  };
}

/** Image pixels → screen (viewport-relative CSS pixels). */
export function toScreen(current: Transform, point: Point): Point {
  return {
    x: point.x * current.scale + current.tx,
    y: point.y * current.scale + current.ty,
  };
}

/** A length in screen pixels, expressed in image pixels. For hit slop. */
export function screenToImageLength(current: Transform, length: number): number {
  return length / current.scale;
}

/** The CSS transform string for a layer that renders in image coordinates. */
export function cssTransform(current: Transform): string {
  return `translate(${current.tx}px, ${current.ty}px) scale(${current.scale})`;
}

/** The SVG viewBox showing the image region currently on screen. */
export function viewBox(current: Transform, viewport: Size): string {
  const topLeft = toImage(current, { x: 0, y: 0 });
  return [
    topLeft.x,
    topLeft.y,
    viewport.width / current.scale,
    viewport.height / current.scale,
  ].join(" ");
}

/** Percentage for the zoom readout. Rounded, never "99%" when it is 100%. */
export function zoomPercent(current: Transform): number {
  return Math.round(current.scale * 100);
}
