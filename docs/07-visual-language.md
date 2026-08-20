# 07 — visual language

The style catalog. Intended as input to design work, not output from it.

## What kind of interface this is

A dense professional tool, in the lineage of Lightroom, Capture One and
DaVinci Resolve rather than a web dashboard. Specifically:

- **Long sessions.** Reviewing a few hundred frames, watching a loss
  curve for an hour. Low-fatigue, low-chrome.
- **Keyboard-first.** The review pass is a repetitive loop; the mouse is
  for drawing boxes and nothing else.
- **Information-dense.** Whitespace is not the goal. Fitting the filmstrip,
  the image, and the inspector on one screen is.
- **Viewed over an SSH tunnel from a laptop.** Payload size is a design
  constraint, not an afterthought. See
  [10 — graphics stack](10-graphics-stack.md#bandwidth-budget).

## The rule that drives the palette

**The interface surrounds photographs the user is judging.** Any colour
cast in the chrome biases perception of skin tone, white balance and
contrast in the image next to it. This is why every serious image tool is
a neutral dark grey.

So: **all surfaces are true neutral — R = G = B, no blue tint, no warm
tint.** This is not a stylistic preference and should not be "improved"
into a fashionable near-black-blue.

Dark is the primary theme. A light theme is secondary and lower priority;
if built, it holds the same neutrality rule.

## Colour

### Surfaces (dark, primary)

| Token | Value | Use |
|---|---|---|
| `bg-void` | `#101010` | Image viewport surround, canvas backdrop |
| `bg-app` | `#181818` | Application background |
| `bg-panel` | `#202020` | Panels, sidebars, toolbars |
| `bg-raised` | `#282828` | Cards, popovers, menus, table headers |
| `bg-input` | `#161616` | Inputs, wells, recessed areas |
| `border` | `#333333` | Standard dividers |
| `border-strong` | `#454545` | Focused/active edges, panel separation |

### Text

| Token | Value | Use |
|---|---|---|
| `text-primary` | `#EDEDED` | Body, values, labels |
| `text-secondary` | `#A0A0A0` | Supporting text, column headers |
| `text-tertiary` | `#6E6E6E` | Hints, placeholders, disabled |

### Interactive

| Token | Value | Use |
|---|---|---|
| `accent` | `#4A9EFF` | Focus rings, selection, primary actions, links |
| `accent-hover` | `#6BB0FF` | Hover state |
| `accent-muted` | `#1E3A5C` | Selected row fill, accent backgrounds |

One accent only. Resist a second brand colour — every additional hue
competes with the state colours and with the imagery.

### State

Never the sole carrier of meaning; always paired with an icon or a label.

| Token | Value | Meaning |
|---|---|---|
| `state-running` | `#4A9EFF` | Training in progress |
| `state-success` | `#3FB950` | Complete, valid, in sync |
| `state-warning` | `#D29922` | Drift, version skew, unreviewed |
| `state-error` | `#F85149` | Failed, missing, conflict |
| `state-queued` | `#8B949E` | Waiting |
| `state-paused` | `#A371F7` | Paused |

### Overlay colours

These sit **on top of arbitrary photography** — gym interiors, skin,
hardwood, gi fabric, stage lighting. They are chosen to be rare in that
content so they stay legible:

| Token | Value | Use |
|---|---|---|
| `overlay-mask` | `#FF00AA` @ 35% | Mask region fill |
| `box-detected` | `#00E5FF` | Machine-detected face box |
| `box-manual` | `#FFD000` | Hand-drawn box |
| `box-selected` | `#FFFFFF` | Selected box, 2px + corner handles |
| `box-shadow` | `#000000` @ 60% | 1px outer stroke under every box line |

Magenta and cyan are near-absent from skin tones, wood, foliage and
neutral studio backdrops, which is exactly why they are used here and
nowhere else in the interface.

The 1px dark outer stroke under every overlay line is not decoration — it
is what keeps a cyan box visible over a blown-out white wall.

### Data visualisation

| Token | Value | Use |
|---|---|---|
| `series-loss` | `#6E6E6E` | Raw loss, low emphasis |
| `series-ema` | `#4A9EFF` | EMA, high emphasis |
| `series-lr` | `#A371F7` | Learning rate, secondary axis |
| `series-grid` | `#2A2A2A` | Gridlines |

Raw loss is deliberately quieter than its own smoothing — the EMA is the
line being read, the raw series is context.

### Contrast

Body text ≥ 4.5:1 against its surface. Secondary text and disabled states
≥ 3:1 — below AA for body but acceptable for genuinely non-essential
text, and the density of the tool depends on it. State colours are all
≥ 3:1 on `bg-panel`. Overlay colours are exempt (they sit on photographs,
where contrast is not controllable) which is what the dark outer stroke
compensates for.

## Typography

### Families

| Role | Face | Weights | Notes |
|---|---|---|---|
| UI | **Inter** | 400, 500, 600 | Variable, excellent at 12–14px, real tabular numerals |
| Mono | **JetBrains Mono** | 400, 700 | Logs, paths, hashes, config |

Both **self-hosted with the daemon**. No CDN: lab nodes may be offline or
air-gapped, and font loading over an SSH tunnel should not be a thing
that can fail. Fallback stacks are declared but should never be reached.

JetBrains Mono is chosen over the usual candidates for disambiguation —
`0/O`, `1/l/I`, `5/S` are visually distinct, which matters when the text
is a checkpoint hash or a Linux path being compared by eye.

### Tabular numerals are mandatory

Any number that updates in place — loss, step count, VRAM, ETA, queue
depth, progress percentages — uses `font-variant-numeric: tabular-nums`.
Proportional digits cause horizontal jitter on every tick, which over a
long training run is genuinely unpleasant.

This also applies to every numeric table column, so values align on the
decimal.

### Scale

Base is **13px**, not 16px. This is a tool, not a document.

| Token | Size / line-height | Use |
|---|---|---|
| `text-xs` | 11 / 1.4 | Chips, badges, dense metadata |
| `text-sm` | 12 / 1.45 | Table cells, secondary labels, logs |
| `text-base` | 13 / 1.45 | Body, inputs, most UI |
| `text-md` | 14 / 1.4 | Emphasised values, section labels |
| `text-lg` | 16 / 1.35 | Panel titles |
| `text-xl` | 20 / 1.3 | Screen titles |
| `text-2xl` | 24 / 1.25 | Stat tile values |

Weight carries hierarchy more than size: 600 for headings and emphasis,
500 for labels and buttons, 400 for everything else. Letter-spacing
`-0.01em` at 16px and above; `0` below.

## Spacing and density

4px base unit. Scale: `4, 8, 12, 16, 24, 32, 48`.

Two densities, switchable:

| | Compact (default) | Comfortable |
|---|---|---|
| Table row | 28px | 36px |
| Control height | 28px | 32px |
| Panel padding | 12px | 16px |
| Gap between controls | 8px | 12px |

Compact is the default because the review screen needs the filmstrip, the
image and the inspector visible at once on a laptop display.

Minimum hit target is 28px in dense zones and 32px elsewhere — below the
44px touch guidance, which is the correct trade for a pointer-driven pro
tool. **Tablet mode raises every target to 44px** and switches to
Comfortable automatically.

## Elevation and shape

**Borders, not shadows.** Drop shadows read as grey haze on a dark
neutral surface and muddy the very neutrality the palette exists to
protect. Depth comes from the surface ramp (`bg-panel` → `bg-raised`)
plus a `border` line.

The single exception: popovers, menus and modals get
`0 8px 24px rgba(0,0,0,0.5)` to separate them from the content they cover.

Radii are tight — this is a technical instrument:

| Token | Value | Use |
|---|---|---|
| `radius-sm` | 3px | Chips, badges, table cells |
| `radius-md` | 4px | Buttons, inputs, menu items |
| `radius-lg` | 6px | Panels, cards, modals |

No radius on the image viewport or the canvas. Ever — a rounded corner on
a photograph crops the photograph.

## Focus

Keyboard focus must be unmistakable, because most of this interface is
driven without a mouse:

```
outline: 2px solid var(--accent);
outline-offset: 1px;
```

Never removed, never replaced by a colour change alone. Focus-visible
semantics so it does not fire on mouse clicks. Focus order follows visual
order; the review screen manages focus explicitly between filmstrip,
canvas and inspector.

## Iconography

**Lucide**, or an equivalent single-weight line set. 1.5px stroke,
16px in dense UI, 20px in toolbars, 24px in empty states. Self-hosted as
an inline sprite, never an icon font.

Icons never appear without a label or a tooltip, except in toolbars where
the same six icons repeat on every screen and become learned.

## Motion

Minimal and never in the input path.

| Change | Duration | Easing |
|---|---|---|
| Hover, focus, colour | 120ms | `ease-out` |
| Panel/drawer open | 180ms | `cubic-bezier(0.2, 0, 0, 1)` |
| Toast in/out | 150ms | `ease-out` |
| Box drag, zoom, pan | **0ms** | none — follows the pointer exactly |

Direct manipulation is never animated. A box being dragged is under the
cursor, full stop.

`prefers-reduced-motion: reduce` drops everything to 0ms except opacity
fades.

Progress indicators reflect real progress. No indeterminate bar standing
in for a determinate operation — the API reports step counts, so use them.

## Additions from screen pass 02

The token set above was written in the abstract. Composing the four
screens found four things it was missing. These are now in
`design/tokens.css` and are part of the system:

| Token | Value | Why it was needed |
|---|---|---|
| `bg-subtle` | `#1C1C1C` | A rung between `bg-app` and `bg-panel`, for nested rows and the filmstrip ground |
| `bg-hover` | `#232323` | A rung between `bg-panel` and `bg-raised`, for row hover and group headers |
| `border-subtle` | `#262626` | `border` at `#333333` is too heavy as a row rule at 28px row height |
| `text-on-accent` | `#101010` | Never specified what colour sits on an accent fill |

The five-step surface ramp was simply not enough for dense compositions.
That is a normal finding and the ramp is now seven.

**State surfaces** are a partial addition. The state palette defined
foregrounds but no tinted fills or borders to sit behind them, and the
screens improvised several. `state-warning-bg/border` and
`state-error-bg/border` are adopted as drawn; the remaining tints
(success, running, paused, and several depths of amber) still need
normalising into one consistent set rather than being reverse-engineered
from usage.

Note the neutrality rule does **not** extend to these. A warning row fill
should be tinted; it is a filmstrip row, not the surround of the image
being judged. Neutrality binds the surfaces adjacent to photographic
content, which is where it actually matters.

## Asset constraints

- **Everything self-hosted.** Fonts, icons, styles, scripts. The daemon
  serves the client; there is no network beyond it.
- **No runtime CSS-in-JS.** Tokens are CSS custom properties so the theme
  is inspectable and swappable without a rebuild.
- **Total client payload budget: under 500KB** compressed, excluding
  imagery. It is loaded over a tunnel, sometimes on a hotel connection.
