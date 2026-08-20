# 00 — build handoff

Read this first if you are starting a build session. Everything before
this point was planning; this is the state it left behind and the first
thing to do.

## Where things stand

**Design phase complete. No code exists yet.** The repository holds
2,100 lines of specification across `docs/`, plus a rendered component
catalog and screen compositions in `design/`.

Nothing has been built. `src/` does not exist. P0 is greenfield.

The v1 application at `D:\Projects_26\AI_Image_Trainer` is still the
working tool and stays that way until this is genuinely ahead.

## Settled — do not relitigate

| | |
|---|---|
| Scope | Full rewrite of the tooling around training |
| Kohya / sd-scripts | Dropped. Evidence in [01](01-v1-audit.md#dropped-kohya--sd-scripts) |
| Structure | Headless core → per-node daemon → clients |
| Backends | ai-toolkit (FLUX + Krea 2 as one) and Klein. Two, not four |
| Remote | REST + SSE, localhost-bound, driven over SSH tunnels |
| Fleet | Client-side aggregation, no coordinator |
| Dataset storage | Node-local; manifest diff over rsync or tar |
| UI | Browser client served by the daemon. **PyQt6 is not carried forward** |
| Face detection | OpenCV YuNet behind a pluggable `Detector` interface |
| Mask delivery | ai-toolkit `mask_path` loss masking, not baked-in pixels |

## Open — decide before the phase that needs it

Four client-stack choices, all deferrable to P6, listed in
[10](10-graphics-stack.md#what-to-decide-before-building): framework,
virtualizer, chart approach, icon set. The log viewer sets the
virtualizer bar (100k lines, follow-tail, filter that changes row
heights) and the streaming-append requirement rules out most declarative
chart libraries.

Three product questions in [05](05-roadmap.md#open-questions), of which
the one that changes near-term work is: **Klein or Krea 2 first?** The v1
README says Klein, every recent config says Krea 2. That decides whether
P4 or P5 comes first.

One design leftover: the state-surface tints are not normalised. Success,
running, paused and several amber depths are still floating as literals
in the screen compositions rather than tokens. Not blocking; fix before
P6.

## Start here: P0

Scaffold only. Definition of done:

1. `pyproject.toml`, package `fluxkrea` under `src/`.
2. `core/paths.py` — every path resolved in one place, `pathlib`
   throughout, no drive-letter or separator assumptions.
3. `core/config.py` — dataclass-backed, one file, precedence
   `defaults < file < env < flags`. Secrets from env or keyring, never
   the file.
4. `core/events.py` — `Progress`, `Log`, `LossPoint`, `Finished` as
   frozen dataclasses, plus the `Emitter` type.
5. `pytest` harness.
6. **The guard test**: walk `core/`, fail on any UI toolkit import. This
   is the rule the whole architecture rests on and it should exist before
   there is anything to guard.
7. Green on Windows and Linux both.

Then P1 (dataset core) and P2 (face masking), both headless. **P2 is
where the project pays for itself** — a `masks/` folder that Krea 2
trains against, before any UI exists.

## Things to carry in your head

- **`klein_trainer/` gets ported, not improved.** 4,165 lines of working
  ML code. Move it, wrap it, touch it afterwards. Cleaning it up during
  the move is the most likely way to break training.
- **Every dataset operation goes through `DatasetItem`.** The v1 bug that
  started all this was three hand-rolled copies of sidecar handling. One
  place knows a bundle is image + caption + mask.
- **Masks resample with NEAREST**, never Lanczos. A grey pixel in a mask
  is a partial loss weight — a soft leak of the region being excluded.
- **Anything the UI can do, the API can do.** No feature reachable only
  by clicking.
- **v1's fixes are already made** (commit `d1890ce` in the v1 repo): EXIF
  orientation, file handles, two-phase rename with rollback, subset
  renumbering. Carry them forward rather than rediscovering them; the
  list is in [01](01-v1-audit.md#dataset-side-rot-fixed-in-d1890ce).

## Design artifacts

| File | Status |
|---|---|
| `design/tokens.css` | **Authoritative for token values.** 119 lines, drop-in |
| `design/FluxKrea Design Catalog.dc.html` | 56 component specimens + props reference |
| `design/FluxKrea Screens.dc.html` | Four screens, 1280 variant, edge states |
| `design/brief-02-screens.md` | The brief that produced the screens |

The `.dc.html` files need `support.js` beside them; serve the folder over
HTTP rather than opening from disk.

Two notes on the design output. The catalog HTML loads fonts from the
Google Fonts CDN — the client must self-host, and `tokens.css` carries
the `@font-face` shape needed. And the props reference tables in the
catalog are an implementation contract worth reading before writing any
component; they encode the rules, not just the shapes.
