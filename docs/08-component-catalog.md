# 08 — component catalog

Every element the client needs. Tokens referenced here are defined in
[07 — visual language](07-visual-language.md).

Grouped as primitives (generic), data display, feedback, navigation, and
domain components — the last group being the ones that do not exist off
the shelf and carry the real work.

---

## Primitives

**Button** — variants `primary` (accent fill), `secondary` (bordered,
transparent), `ghost` (no border until hover), `danger` (error fill).
Sizes `sm` 24px / `md` 28px / `lg` 32px. States: default, hover, active,
focus-visible, disabled, loading (spinner replaces icon, label stays,
width locked to prevent reflow).

**Icon button** — square, 28px, tooltip mandatory. Toggle variant holds a
pressed state with `accent-muted` fill.

**Button group** — segmented, single-select. Used for view mode, density,
overlay toggles.

**Text input** — 28px, `bg-input`, `border`, accent border on focus.
Variants: with prefix icon, with unit suffix (`px`, `×`, `steps`), with
inline validation message below. Monospace variant for paths and hashes.

**Number input** — tabular numerals, stepper affordances, drag-to-scrub
on the label (a pro-tool convention worth keeping for expansion factor
and feather radius). Clamps to min/max, shows the clamp rather than
silently correcting.

**Slider + number** — always paired, never a bare slider. The slider is
for feel, the number is for precision and for typing an exact value.
Required for mask opacity, expansion factor, feather, LoRA rank.

**Select / dropdown** — native-feeling, keyboard navigable, type-ahead.
Grouped options with section headers (models grouped by family).

**Combobox** — filterable select for long lists: model paths, node names,
dataset names.

**Checkbox / radio / switch** — switch for immediate-effect settings
(overlay on/off), checkbox for form state committed on save. Do not mix
the metaphors.

**Chip / tag** — small, `radius-sm`. Variants: static label, removable
(× affordance), toggleable filter. Filter chips carry a count
(`unreviewed 26`).

**Tooltip** — 200ms delay, `bg-raised`, follows the pointer for canvas
elements and anchors for UI elements. Never contains interactive content.

**Popover** — anchored panel for compound controls: detector settings,
column picker, export options.

**Menu / context menu** — right-click on table rows, filmstrip items and
canvas boxes. Keyboard navigable, with shortcut hints right-aligned.

**Segmented progress bar** — determinate only, tabular percentage
alongside, indeterminate variant reserved for genuinely unknown duration.

**Spinner** — 14px inline, 24px block. Never a full-screen blocker.

---

## Data display

**Table** — the workhorse. Requirements: virtualized rows, sortable
columns, resizable columns, sticky header, multi-select with
shift/ctrl-click, row context menu, keyboard row navigation, per-column
alignment (numbers right, tabular), configurable column visibility, and a
compact 28px row height. Zebra striping is *off* — use hover and
selection instead, since striping fights with per-row status colour.

**Key–value list** — two-column definition list for metadata panels.
Labels `text-secondary` right-aligned, values `text-primary` left-aligned,
monospace where the value is a path or hash.

**Stat tile** — large tabular number, small label, optional delta and
sparkline. Used for step count, loss, VRAM, ETA, queue depth. Fixed
width so a row of them does not reflow as values change digits.

**Status pill** — icon + label + state colour. The single canonical way
state is shown, everywhere: jobs, nodes, dataset items, validation.

**Badge / count** — small numeric badge on tabs and filter chips.

**Progress row** — label, bar, `184/210`, ETA. Composed, not bespoke per
screen.

**Sparkline** — 60×16, no axes, single series. Inside stat tiles and node
rows.

**Empty state** — icon, one-line explanation, one primary action. Written
for the specific situation, never "No data".

**Diff list** — for dataset drift: rows of added / changed / missing with
per-item size, grouped and collapsible.

---

## Feedback

**Toast** — bottom-right stack, 4s auto-dismiss for success, sticky for
error with a Retry action and a Details expander. Never used for anything
the user must act on.

**Inline alert** — `info` / `warning` / `error`, inside the panel it
concerns. Preferred over toasts for anything contextual.

**Confirm dialog** — title, consequence sentence, cancel + action.
**Destructive variant** names the object in the button (`Delete 210
masks`), not a bare "Confirm", and requires the typed dataset name for
anything irreversible across a whole dataset.

**Skeleton** — for thumbnail grids and tables only, matching final
dimensions so nothing shifts on load. Everything else uses a spinner.

**Connection banner** — persistent top bar when a node is unreachable or
the SSE stream has dropped, with reconnect countdown. This matters more
than usual: the connection is an SSH tunnel and it *will* drop when the
laptop sleeps.

---

## Navigation

**App shell** — top bar (node selector, global search, connection state,
settings) + left icon rail (Datasets, Review, Training, Fleet) + content
region. The rail is icons-only at 48px with tooltips, expandable to 180px
with labels.

**Tabs** — within a screen only, never as primary navigation.

**Breadcrumb** — for path browsing and dataset drill-down. Segments are
clickable, middle segments collapse to `…` under width pressure.

**Command palette** — `Ctrl/Cmd+K`. Actions, datasets, nodes, jobs. This
is the escape hatch that lets the whole tool be driven without learning
where anything lives, and it should be built early rather than last.

**Path browser** — modal, two-pane (tree + listing), breadcrumb,
keyboard-navigable, scoped to configured roots. **This replaces the
native file dialog** that the web client gives up, so it has to be
genuinely good: type-ahead, recent paths, favourites, and a manual path
entry field that accepts a pasted absolute path.

---

## Domain components

The ones that carry the actual work.

### Image viewport

Zoom/pan surface hosting the image, mask overlay and box layer.
Requirements: fit-to-window, 1:1, zoom to point under cursor, spacebar
pan, pixel-snapping at ≥100% so mask edges are not resampled visually,
and a persistent zoom indicator. Background is `bg-void`, no border, no
radius.

### Box overlay editor

The centrepiece. Over the image viewport:

- Draw by drag on empty space; creates a `manual` box.
- Select by click; multi-select by shift-click or marquee.
- Move by drag; resize by 8 corner/edge handles; handles stay a constant
  screen size regardless of zoom.
- Delete by `Del`/`Backspace`.
- Nudge by arrow keys (1px, 10px with shift).
- Constrain to image bounds; a box cannot be dragged off-canvas.
- Colour by source (`box-detected` cyan, `box-manual` amber,
  `box-selected` white) with the dark outer stroke from doc 07.
- Live readout of the selected box's pixel geometry.
- **Undo/redo, scoped to the current image**, minimum 50 steps.

Boxes are *source geometry*. The mask preview is derived from them
through expansion and feather, and re-derives live as those change.

### Mask overlay

Toggleable render of the derived mask over the image: `overlay-mask`
magenta at adjustable opacity (0–100%, default 35%). Three view modes
cycled by one key: **off**, **overlay**, **isolate** (mask only, image
hidden) — isolate being how you verify coverage without the photograph
distracting from it.

### Filmstrip / item list

Virtualized vertical list of dataset items. Each row: thumbnail, stem,
status pills (reviewed / unreviewed / no-detections / box count).
Sortable and filterable; **defaults to sorting zero-detection items
first**, because that is where misses hide. Keyboard navigable
independently of canvas focus.

### Thumbnail grid

Virtualized, adjustable cell size, lazy-loaded server-generated
thumbnails. Overlays per cell: quality rating, caption-present indicator,
mask-present indicator, selection checkbox. Multi-select with rubber-band.

### Caption editor

Textarea with token/character count, per-item, with "apply to selection"
and "append to all" affordances. Dirty-state indicator; explicit save.

### Loss chart

Streaming line chart. Raw loss in `series-loss`, EMA in `series-ema`, LR
on a secondary axis. Requirements: append without full re-render,
decimation above a few thousand points, log-scale toggle, hover crosshair
with tabular readout, brush-to-zoom, and **outlier markers** that link
back to the offending training image.

### Sample strip

Horizontal timeline of generated samples grouped by step, click to open
full size, compare-at-step against an earlier sample. Lazy-loaded.

### Log stream viewer

Virtualized, monospace, follow-tail with an auto-disable on manual
scroll-up and a "jump to latest" affordance. Level filtering, text
filter, ANSI colour passthrough, copy-selection. Must stay responsive at
tens of thousands of lines.

### GPU meter

Per-device: name, VRAM used/total as a bar, utilisation, temperature.
Compact variant for node rows, expanded for the node detail panel.

### Node card / node row

Hostname, status pill, GPU meters, running job, queue depth, torch/CUDA
version, last-seen. Warning treatment when the version differs from the
fleet majority — the Blackwell `sm_120` requirement makes silent skew
expensive.

### Job queue row

Position, dataset, model, backend, status pill, progress row, ETA,
actions (pause, resume, cancel). Drag to reorder pending jobs.

### RunSpec form

The training configuration form. Grouped sections (model, dataset,
network, schedule, sampling, output), presets dropdown, live-derived
readouts (total steps from images × repeats × epochs), inline validation,
and a diff view against the selected preset so it is obvious what has
been changed. Save-as-preset.

### Dataset placement panel

Which nodes hold this dataset, manifest agreement per node, per-node
push/pull actions, and a sidecars-only push affordance surfaced
prominently — it is the common case in the mask iteration loop.

### Validation report

Grouped, collapsible list of problems from `GET /datasets/{id}/validate`:
missing captions, orphaned sidecars, missing masks, dimension mismatches.
Each row links to the offending item. Counts by severity at the top.
