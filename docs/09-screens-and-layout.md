# 09 — screens and layout

How the elements are arranged. Components referenced here are specified in
[08 — component catalog](08-component-catalog.md).

## App shell

```
┌────────────────────────────────────────────────────────────────┐
│ [node ▾]  FluxKrea 26          ⌘K search       ● connected  ⚙ │ 44px
├────┬───────────────────────────────────────────────────────────┤
│ ▣  │                                                           │
│ ◧  │                    screen content                         │
│ ⏵  │                                                           │
│ ⛓  │                                                           │
│    │                                                           │
│ 48 │                                                           │
└────┴───────────────────────────────────────────────────────────┘
```

- **Top bar, 44px, fixed.** Node selector at far left — everything the
  screen shows is scoped to it, so it reads first. Connection state at
  far right, because over an SSH tunnel it is a thing you check.
- **Left rail, 48px, fixed.** Datasets / Review / Training / Fleet.
  Expandable to 180px with labels, state persisted.
- **Content region** owns its own scrolling. The shell never scrolls.

Minimum supported viewport is **1280×720** — a laptop over a tunnel. Below
1440px wide, secondary inspector panels collapse to a toggle.

---

## Dataset gallery

```
┌──────────────────────────────────────────┬─────────────────┐
│ poses_v3   210 items   ⚠ 6 issues        │                 │
│ [scan] [validate] [mask…] [caption…]     │   inspector     │
├──────────────────────────────────────────┤                 │
│ [all ▾] [unmasked 26] [no caption 3]     │   thumbnail     │
├──────────────────────────────────────────┤   stem          │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢                        │   dimensions    │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢    thumbnail grid      │   caption ────  │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢    (virtualized)       │   │           │ │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢                        │   └───────────┘ │
│                                          │   quality       │
│                                          │   mask ✓        │
└──────────────────────────────────────────┴─────────────────┘
   flexible                                    320px fixed
```

Header carries dataset identity and the destructive-ish batch actions.
Filter chips below it are the primary navigation of a large dataset and
carry live counts. The inspector shows one item, or aggregate stats plus
batch caption tools when multiple are selected.

Cell size is user-adjustable (`[` and `]`). Selection is the input to
every batch operation, so the selection count is always visible in the
header.

---

## Mask review

The screen the whole project exists for. Everything else can be
mediocre; this one cannot.

```
┌─────────────────────────────────────────────────────────────────┐
│ poses_v3 · review      184/210 reviewed · 6 no detections       │ 40
│ [unreviewed] [no detections] [all]                              │
├──────────┬──────────────────────────────────────┬───────────────┤
│          │                                      │ BOXES         │
│ ▣ 001 ✓  │                                      │ ┌───────────┐ │
│ ▣ 002 ✓  │                                      │ │ 1 detected│ │
│ ▣ 003 ⚠  │         image viewport               │ │ 2 manual  │ │
│ ▣ 004    │         + mask overlay               │ └───────────┘ │
│ ▣ 005    │         + box layer                  │ x 412  w  96  │
│ ▣ 006    │                                      │ y  88  h 128  │
│   …      │                                      │               │
│          │                                      │ MASK          │
│ filmstrip│                                      │ expand  1.6×  │
│          │                                      │ feather 12px  │
│          │                                      │ opacity 35%   │
│          │                                      │               │
│          │                                      │ DETECTOR      │
│          │                                      │ yunet ▾       │
│          │                                      │ [re-detect]   │
├──────────┼──────────────────────────────────────┴───────────────┤
│          │ [fit] [1:1] 68%   [M]ask [D]etected  ◀ 184/210 ▶     │ 36
└──────────┴──────────────────────────────────────────────────────┘
   220px              flexible, centred              260px
```

Arrangement rationale:

- **The image gets every pixel that is left.** Side panels are as narrow
  as they can be and still be usable, because judging whether a face is
  fully covered is a visual-acuity task.
- **Filmstrip left, inspector right.** Progression is a left-to-right
  reading order: what to look at → the thing → what to change about it.
- **Status is in the filmstrip, not the canvas.** Nothing overlays the
  image except the mask and the boxes.
- **The bottom bar holds view state**, not actions — zoom, overlay
  toggles, position in the set. It never changes the data.
- **Zero-detection items sort first** by default and are marked `⚠`.
  A miss is the failure mode that defeats the whole feature, so the
  interface pushes them at you rather than waiting to be asked.

### Keyboard map

The review pass must be completable without the mouse except for drawing.

| Key | Action |
|---|---|
| `J` / `↓` | Next item |
| `K` / `↑` | Previous item |
| `Space` | Mark reviewed, advance |
| `Shift+Space` | Mark reviewed, do not advance |
| `B` | New box mode (then drag) |
| `Del` / `Backspace` | Delete selected box |
| `Tab` | Cycle boxes on this image |
| `←↑→↓` | Nudge selected box (1px; 10px with `Shift`) |
| `M` | Cycle mask view: off → overlay → isolate |
| `D` | Toggle detected boxes |
| `0` | Fit to window |
| `1` | 100% zoom |
| `+` / `-` | Zoom in / out |
| `Space` (hold) | Pan |
| `R` | Re-detect this image |
| `Shift+R` | Re-detect all unreviewed |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo box edit |
| `F` | Focus filter chips |
| `?` | Shortcut overlay |

`Space` is doing double duty as both "mark reviewed" and "hold to pan" —
tap versus hold. If that proves ambiguous in practice, panning moves to
middle-drag and `H`.

---

## Training monitor

```
┌────────────────────────────────────────────────────────────────┐
│ Blizzard_krea2 · olympus-2 · ● running    [pause] [stop]        │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│ step     │ loss     │ ema      │ vram     │ eta                │
│ 2,340    │ 0.0412   │ 0.0388   │ 21.4/24  │ 1h 12m             │  stat tiles
│ /4,700   │ ▁▂▁▃▂▁   │ ↘ steady │          │                    │
├──────────┴──────────┴──────────┴──────────┴────────────────────┤
│                                                                │
│                     loss chart (raw + EMA + LR)                │ 40%
│                                                                │
├────────────────────────────────────────┬───────────────────────┤
│  samples strip (by step)               │  log stream           │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ →                     │  (virtualized, tail)  │ 35%
└────────────────────────────────────────┴───────────────────────┘
```

Stat tiles first because they answer "is it healthy" in one glance. The
chart is the largest single element since it is what gets watched. Samples
and logs share the lower band; either can be maximised.

Outlier markers on the chart link to the training image responsible —
that connection is the most useful thing Klein's analytics produce and it
should be one click, not a separate panel.

Live config controls (LR, sampling cadence) live in a popover off the
header, not inline. They are rare and consequential; they should not be
adjacent to the mouse's resting position.

---

## Fleet

A **table, not a card grid.** Cards look better in a screenshot; a table
lets you compare eight nodes down a column, which is the actual task.

| node | status | GPU / VRAM | job | queue | datasets | torch / CUDA |
|---|---|---|---|---|---|---|
| olympus-1 | ● running | Blackwell 4000 ▓▓▓▓░ 21/24 | Blizzard 2340/4700 | 2 | 4 ✓ | 2.6.0 / 12.6 |
| olympus-2 | ○ idle | Blackwell 4000 ▓░░░░ 2/24 | — | 0 | 3 ⚠ drift | 2.6.0 / 12.6 |
| olympus-3 | ✕ unreachable | — | — | — | — | — |

Row click opens a node detail panel: full GPU meters, disk, dataset
placement, job history. Version cells are warning-treated when they
differ from the fleet majority.

Unreachable nodes stay in the table as a greyed row with a last-seen
time. They do not disappear — a missing row reads as "I forgot to
configure it", a greyed row reads as "it is down".

---

## Job queue

Cross-node list, filterable by node and status, drag to reorder pending
jobs within a node. Submitting is a modal over the RunSpec form with the
target node selected at the top, because placement is explicit.

---

## Settings

Plain two-column form: sections left, content right. Covers node
connections and tunnel ports, dataset roots, detector defaults, captioner
backends and keys, appearance (theme, density), keyboard shortcuts.

Secrets never render their value — status only (`configured` / `not
set`) with a replace action, matching the "no secrets in config files"
rule from [06](06-remote-and-fleet.md#security).

---

## Responsive behaviour

| Width | Behaviour |
|---|---|
| ≥1600px | Everything visible; review inspector at 300px |
| 1280–1600px | Default layout as drawn |
| 1024–1280px | Inspector panels become toggleable overlays |
| <1024px | Single-column; review screen is read-only, no box editing |

Box editing is deliberately disabled below 1024px rather than
degraded. Precise geometry work on a cramped canvas produces bad masks,
and a bad mask is worse than a deferred one.

Tablet mode (touch detected) raises hit targets to 44px, switches to
Comfortable density, and replaces box handles with larger touch grips.
