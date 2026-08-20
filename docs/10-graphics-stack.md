# 10 — graphics stack

The rendering capabilities the client needs, and the constraints they sit
under. This is the "what tools are required" half of the design catalog —
what has to be built or chosen before the screens in
[09](09-screens-and-layout.md) can exist.

## Layered rendering model

The image viewport is **three layers sharing one transform**, not one
technology:

```
┌─ container (owns pan/zoom transform) ──────────┐
│  <canvas>   image + derived mask compositing   │  raster
│  <svg>      boxes, handles, marquee            │  vector
│  <div>      HTML controls, readouts, tooltips  │  DOM
└────────────────────────────────────────────────┘
```

Why split rather than do everything on canvas:

- **The image and mask belong on canvas.** Compositing a magenta mask at
  35% over a photograph, and the isolate view, are pixel operations. Doing
  them in the DOM means filter hacks that behave differently per browser.
- **Boxes belong in SVG.** There are rarely more than a handful per image,
  and SVG gives hit-testing, hover, focus, keyboard targets and crisp
  handles for free. Reimplementing hit-testing and focus rings on canvas
  is work with no payoff at this element count.
- **A single shared transform** keeps them registered. One `{scale, tx,
  ty}` object drives the canvas draw and the SVG `viewBox`; there is
  never a second source of truth about where the image is.

If box counts ever reach the hundreds — they will not, for faces — the
SVG layer swaps to canvas without touching the other two.

## Viewport / transform manager

A small, well-tested module, since everything visual depends on it:

- `fit()`, `actualSize()`, `zoomAtPoint(clientX, clientY, delta)`,
  `pan(dx, dy)`.
- Zoom about the cursor, not the centre. Anything else feels broken.
- Clamp range 5%–1600%; snap to 100% within a tolerance.
- **Pixel snapping at ≥100%** so mask edges are shown as they actually
  are, not as bilinear smear. A soft edge in the display that is not in
  the data will send someone chasing a bug that does not exist.
- Screen↔image coordinate conversion, used by every box operation.
- Handle sizes are computed in *screen* space so they stay 8px at any
  zoom.

## Interaction machinery

- **Pointer Events**, not mouse events — one code path for mouse, pen and
  touch, with pointer capture during drags so a fast drag off-element
  does not drop the gesture.
- **Modal drag state machine**: idle → drawing / moving / resizing /
  marqueeing → commit. Explicit states, because overlapping ad-hoc drag
  handlers is where this kind of editor rots.
- **Undo/redo via a command stack**, scoped per image, 50 steps minimum.
  Every box mutation is a command with `do`/`undo`. Retrofitting this is
  painful; it goes in from the start.
- **Keyboard shortcut registry with scopes** (global / filmstrip /
  canvas / inspector / modal). One place that knows all bindings, so the
  `?` overlay and the settings screen are generated rather than
  hand-maintained.

## Virtualization

Three places need it, and all three are load-bearing:

| Surface | Scale | Notes |
|---|---|---|
| Thumbnail grid | 10k+ items | 2D windowing, lazy image loading, cancel loads on fast scroll |
| Filmstrip | 10k+ items | 1D, must support scroll-to-selected from keyboard nav |
| Log stream | 100k+ lines | 1D, variable height, follow-tail, stable scroll anchoring |

The log viewer is the hardest of the three: follow-tail plus
scroll-anchoring plus a text filter that changes row heights. Worth using
a proven virtualizer rather than writing one.

## Thumbnails and image delivery

**Thumbnails are generated server-side by the daemon and cached.** The
client never receives a 2K training image to draw a 160px cell — over a
tunnel that is the difference between a usable grid and an unusable one.

- Sizes: 160px and 480px on the long edge, WebP, quality ~75.
- Generated on scan, cached beside the dataset, regenerated on mtime
  change.
- **Content-addressed URLs**: `/thumb/{digest}.webp`. Immutable
  `Cache-Control`, so re-running a mask pass changes the digest and busts
  the cache automatically, with no invalidation logic anywhere.
- Full-resolution fetches are range-request capable for progressive
  display, and cancelled when the user moves on.
- `createImageBitmap()` for decode off the main thread; the review loop
  must not stutter while stepping through images.

## Mask compositing

The mask shown in review is **derived live from box geometry**, not
fetched — expansion factor and feather change interactively, so a
round-trip per adjustment is not viable.

- Boxes → expanded rects → feathered alpha, drawn to an offscreen canvas
  at image resolution.
- Composited over the image with `globalAlpha` and the magenta fill for
  overlay mode; drawn alone for isolate mode.
- Debounced to animation frames; a slider drag redraws at 60fps on a
  4K image, which means the compose step must not re-decode the image.
- The **exported** mask is produced server-side from the same box
  geometry and parameters, so the client preview and the trained artifact
  cannot diverge. The client's job is to be an accurate preview, not the
  source of the file.

## Charting

Requirements the loss chart imposes, which rule out most convenience
libraries:

- **Streaming append** — new points arrive by SSE at up to a few per
  second over hours. Re-rendering the full series per point is not
  acceptable.
- **Decimation** above ~2,000 visible points (LTTB or similar), so a
  20,000-step run still pans smoothly.
- **Two series plus a secondary axis** (raw loss, EMA, learning rate).
- **Log-scale toggle** on the value axis.
- **Brush-to-zoom** with a reset, and a hover crosshair reading tabular
  values.
- **Custom markers** for outliers that carry a payload (the image id) and
  are clickable.

Either a light charting library that exposes the canvas, or a purpose-built
canvas renderer. A heavyweight declarative chart library will fight every
one of the above.

Sparklines are separate and trivial — inline SVG, no library.

## Live data

- **SSE client** with automatic reconnect, exponential backoff, and
  `Last-Event-ID` backfill so a laptop waking from sleep does not lose
  the middle of a run.
- **Connection state is UI state**, surfaced by the connection banner in
  [08](08-component-catalog.md#feedback). Assume the tunnel drops.
- Optimistic UI for box edits (they are local until saved); never
  optimistic for job control.

## Bandwidth budget

Everything is viewed over an SSH tunnel, sometimes from a hotel.

| Asset | Budget |
|---|---|
| Client bundle (JS+CSS+fonts+icons) | < 500KB compressed |
| Thumbnail, 160px | < 12KB |
| Thumbnail, 480px | < 40KB |
| Full image fetch | on demand only, cancellable |
| SSE event | < 1KB, no embedded images |

Sample images from training are the sneaky one — a run generating a
1024×1024 sample every 400 steps will fill a strip fast. They are served
as thumbnails in the strip and full-size only on click.

## Accessibility

- The canvas is not accessible, so **the box list in the inspector is its
  accessible equivalent** — every box is a focusable list item with its
  geometry as text, and every canvas operation has a keyboard path.
- Focus management on screen and modal transitions; focus never lost to
  `<body>`.
- ARIA live region for status changes that are otherwise only colour
  (job finished, node unreachable).
- No meaning carried by colour alone — status is always icon + label.
- `prefers-reduced-motion` honoured throughout.

## Colour management

Out of scope for v1, and worth stating so it is a decision rather than an
oversight: images are displayed without ICC handling, in whatever the
browser does by default. Training data is judged for content and framing,
not for colour accuracy. If it ever matters, it matters at the point
where masks are being cut against skin tones, and it becomes a real
requirement then.

## What to decide before building

1. **Framework**, if any. The constraint is static assets served by the
   daemon with no build-time API coupling; anything meeting that works.
2. **Virtualizer** — the log viewer sets the bar.
3. **Chart approach** — library-with-canvas-access versus purpose-built.
4. **Icon set** — Lucide assumed in [07](07-visual-language.md#iconography).

These are the only four choices that are expensive to reverse later.
