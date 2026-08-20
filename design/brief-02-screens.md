# Design brief 02 — screen compositions

Second pass. The first produced `FluxKrea Design Catalog.dc.html`: tokens,
54 component specimens, and a props reference. This pass composes those
components into the four screens that matter, and fills three gaps.

## What you are designing for

A LoRA training tool for FLUX.2 Klein and Krea 2. A fleet of Linux boxes
with RTX PRO Blackwell 4000 cards does the training; one laptop drives
them all over SSH tunnels. The client is a browser app served by a daemon
on each node.

The originating feature is **face masking**: pose-reference images get
their faces masked out of the training loss, so a pose LoRA learns bodies
and motion and nothing about faces. Reviewing those masks — a few hundred
martial arts and dance frames, checking that every face is covered — is
the interaction this whole application is built around.

Read `docs/09-screens-and-layout.md` for the layouts, and
`docs/07-visual-language.md` for the rules the catalog already encodes.

## Deliverables

Four screen compositions, one artboard each at **1440×900**:

1. Mask review
2. Training monitor
3. Fleet
4. Dataset gallery

Plus **one variant artboard**: mask review at **1280×720**, the minimum
supported viewport — a laptop over a tunnel, which is the real working
condition.

Plus three component specimens missing from catalog v1, added in the
catalog's existing style: **tabs**, **breadcrumb**, and a **rendered data
table** (the last is specified in the props reference but never drawn,
and the Fleet screen is entirely a table).

## Hard constraints

- **Use the existing tokens. Do not introduce new ones.** They are in
  `design/tokens.css`. If a composition genuinely needs a value that
  isn't there, flag it rather than inventing it.
- **True neutral surfaces, R = G = B.** No blue-black, no warm grey. The
  chrome surrounds photographs being judged and any cast biases
  perception of skin tone. This is the single rule most likely to be
  "improved" by accident.
- **Compact density**: 28px rows and controls, 12px panel padding.
- **Tabular numerals** on every number that updates in place.
- **Borders, not shadows**, except popovers and modals.
- **No radius on the image viewport.** A rounded corner on a photograph
  crops the photograph.
- Fonts: Inter and JetBrains Mono. CDN loading is fine *in the artboard*;
  the shipped client self-hosts, so don't design anything that depends on
  a font that might not load.

## Use realistic content

Perfect placeholder data hides the problems that matter. Populate with
this:

- **Datasets**: `poses_v3` (210 items), `blizzard_char_ref` (48 items),
  `dance_contemporary_2601` (put a long name in at least one place and
  let it truncate).
- **Models**: `krea2_raw.safetensors`, `flux2-klein-9b`.
- **Nodes**: `olympus-1` … `olympus-6`. One idle, one running, **one
  unreachable**, one showing a torch version that differs from the rest
  (`2.5.1 / 12.4` against a fleet on `2.6.0 / 12.6`).
- **A run in progress**: `Blizzard_krea2`, step 2,340 of 4,700, loss
  0.0412, EMA 0.0388, VRAM 21.4/24 GB, ETA 1h 12m.
- **Review progress**: 184 of 210 reviewed, **6 with no detections**.
- **Paths and hashes** in mono where they appear:
  `D:/Projects_26/LoRA_Training_data/Poses/masks`, `ckpt 9f4c1b8e`.

For the image in the review viewport, use a stand-in that behaves like
the real thing: a figure mid-motion, off-centre, with a face at an angle.
Not a centred portrait — the whole problem is faces that are turned away.

## Per screen

### 1. Mask review — the important one

Layout is drawn in doc 09: filmstrip left (~220px), image viewport
centre, inspector right (~260px), 40px header, 36px bottom bar.

**The question this artboard has to answer is proportion.** Is 220 + 260
right, or does the image need more? Show it honestly at 1440 and again at
1280, with a real image in place, and let the proportions be judged.

Must be visible in the composition:

- Mask overlay in **overlay mode** — magenta at 35% over the face — plus
  a small inset or second state showing **isolate mode**, so the three
  view modes read as a set.
- Both box types at once: a cyan detected box and an amber manual box,
  each with the 1px dark outer stroke, and one **selected** in white with
  its 8 handles.
- The selected box's live pixel readout in the inspector.
- Filmstrip rows with all four states: reviewed, unreviewed, **no
  detections (⚠)**, and currently selected.
- **Zero-detection items sorted to the top.** A missed face defeats the
  entire feature, so show how loud that flag is. If it isn't loud enough
  in the composition, make it louder and tell us what you changed.
- Expansion factor, feather and opacity as slider+number pairs.
- Progress readout: `184/210 reviewed · 6 no detections`.

**One open question to take a position on:** the keyboard map has `Space`
doing double duty — tap to mark reviewed, hold to pan. Show whatever
affordance makes that legible in the bottom bar, or propose the
alternative (pan on middle-drag, `H` as the modifier) and show that
instead.

### 2. Training monitor

Stat tile row, large loss chart, then samples strip and log stream
sharing the lower band.

- Chart shows **raw loss quiet, EMA prominent** — the EMA is the line
  being read, raw is context. Learning rate on a secondary axis.
- Include **two outlier markers** on the chart. These link back to the
  training image responsible, which is the most useful thing the
  analytics produce; make that affordance obvious.
- Log stream in follow-tail with the "jump to latest" affordance visible.
- Sample strip with samples at several steps, one selected.
- Show the run **paused** in a second small state, so `state-paused`
  purple appears somewhere.

### 3. Fleet

A dense table, six nodes, sortable columns. This is where the missing
table component gets resolved: sticky header, sort affordance on the
active column, hover row, selected row, and per-row status — all with
zebra striping **off**, as the catalog specifies.

- The **unreachable node stays in the table** as a greyed row with a
  last-seen time. It must not read as "not configured".
- The node on the older torch/CUDA version carries warning treatment.
- GPU meters inline in rows, compact variant.
- One row expanded into the node detail panel.

### 4. Dataset gallery

Virtualized thumbnail grid, filter chips with live counts, inspector on
the right.

- Show a **multi-selection** (say 12 of 210) with the batch state the
  inspector switches to.
- Cell overlays: quality rating, caption-present, mask-present.
- Filter chips carrying counts: `unmasked 26`, `no caption 3`.
- Include the **empty state** for a freshly registered dataset as a small
  second state.

## Edge states — do not skip these

Designs break at the edges, and this tool lives at them. Somewhere in the
set, show:

- **Connection lost.** The persistent banner, with reconnect countdown.
  The tunnel drops every time the laptop sleeps; this is a daily state,
  not an exception.
- **A long dataset name** truncating in a header and a breadcrumb.
- **A validation report with problems** — missing masks, dimension
  mismatch — grouped by severity.
- **A destructive confirm**, with the object named in the button
  (`Delete 210 masks`), not "Confirm".

## What not to do

- No marketing framing, no hero sections, no illustration.
- No second accent colour. One accent, plus the state palette.
- Don't animate anything in the direct-manipulation path; drag and zoom
  follow the pointer exactly.
- Don't design a mobile layout. Below 1024px the review screen is
  deliberately read-only, and that is a product decision, not a gap.
- Don't redesign the token set. If something is wrong, say so.

## How this gets judged

1. Could someone review 210 frames on the mask screen without irritation?
2. Would a missed face be caught, because the interface pushed it forward?
3. Does the fleet table let you compare six nodes down a column at a
   glance, including the broken one?
4. Do the compositions hold at 1280px, or only at 1440?
5. Is every value traceable to a token in `tokens.css`?
