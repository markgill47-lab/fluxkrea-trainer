# FluxKrea Trainer 26

A rewrite of the LoRA training GUI at `D:\Projects_26\AI_Image_Trainer`,
targeting FLUX.2 Klein and Krea 2 across a distributed lab fleet.

**Status: P0–P6 built and green.** The headless core, the per-node daemon
and its API, `fk` as a real API client, the ai-toolkit backend —
**FLUX.2, both Klein sizes and Krea 2 train through one config-driven
class** — and the browser client the daemon serves: dataset gallery, mask
review, training configuration and monitor, settings. The standalone Klein trainer does not
exist yet. v1 stays the working tool until v2 can genuinely replace it.

## Why rewrite

The trigger was a small feature: masking faces in pose-LoRA training
images so the pose LoRA stops fighting the character reference LoRA for
control of the face. Adding it to v1 meant touching three separate
hand-rolled copies of "…and also handle the sidecar file", any one of
which silently breaks the image-to-mask pairing if missed.

That is a symptom. The wider problems, documented in
[docs/01-v1-audit.md](docs/01-v1-audit.md):

- **Backends with no declared interface.** Three of them, each
  re-implementing the same lifecycle with wildly asymmetric features.
- **A 2,420-line god object.** `UIManager` has 66 methods; one of them,
  `setup_config_tab`, is 431 lines.
- **Logic welded to Qt.** Processing functions construct
  `QProgressDialog` and `QMessageBox` directly, so nothing can be tested,
  scripted, or run over SSH.
- **Config sprawl.** Four overlapping stores with unclear precedence, two
  of them gitignored for holding secrets — so the app's real
  configuration cannot be shared between the dev box and the fleet.

## How this gets used

This is not a single-desktop app. The working setup is:

- A **fleet of Linux boxes** with RTX PRO Blackwell 4000 cards, each
  doing the actual training.
- **One laptop**, driving all of them over SSH.
- **Windows and Linux both first-class** — development on Windows,
  production on the fleet.

That shapes the architecture more than anything else: the core is
headless, a daemon on each node exposes a full HTTP API, and every UI is
a client of that API rather than the thing that owns the logic. See
[docs/06-remote-and-fleet.md](docs/06-remote-and-fleet.md).

## Decisions

| Question | Decision |
|---|---|
| Scope | Full rewrite — dataset tools *and* training |
| Kohya / sd-scripts | **Dropped.** Dead since January; SD/SDXL only |
| Structure | Headless core; per-node daemon exposing an HTTP API |
| Remote control | REST + SSE, localhost-bound, driven over SSH tunnels |
| Fleet | Client-side aggregation over a node list. No coordinator |
| Fleet UI | **Not in the node-served client.** It would make every node know about, and reach, every other |
| Captioning | Local by default — JoyCaption in-process, or Ollama. Claude is opt-in |
| Dataset storage | Node-local. Sync by manifest diff over rsync or tar |
| Platforms | Windows and Linux, equally supported |
| Face detection | OpenCV YuNet now, behind a pluggable detector interface |
| Mask delivery | ai-toolkit `mask_path` loss masking, not baked-in pixels |
| UI layer | Browser client served by the daemon. **Qt is not carried forward** |

## Documents

| | |
|---|---|
| **[00 — build handoff](docs/00-build-handoff.md)** | **Start here.** State, settled decisions, and what is next |
| [01 — v1 audit](docs/01-v1-audit.md) | What exists today, what carries over, what dies |
| [02 — architecture](docs/02-architecture.md) | Core/daemon/client split, package layout, backend protocol |
| [03 — dataset model](docs/03-dataset-model.md) | The `DatasetItem` invariant that started all this |
| [04 — face masking](docs/04-face-masking.md) | The originating feature, specified |
| [05 — roadmap](docs/05-roadmap.md) | Phases, port-vs-rewrite, risks, open questions |
| [06 — remote and fleet](docs/06-remote-and-fleet.md) | The API, the daemon, multi-node, security |
| [07 — visual language](docs/07-visual-language.md) | Colour, type, spacing, motion, asset rules |
| [08 — component catalog](docs/08-component-catalog.md) | Every element the client needs |
| [09 — screens and layout](docs/09-screens-and-layout.md) | How it is all arranged, plus keyboard maps |
| [10 — graphics stack](docs/10-graphics-stack.md) | Rendering, canvas, charts, virtualization, budgets |

Docs 07–10 are the design catalog — written to be handed to design work
as input, not produced by it.

## Using it

```bash
pip install -e ".[dev,daemon]"

fk node status                                   # versions, GPUs, detectors
fk prompts list                                  # saved caption prompts
fk dataset caption ./poses --prompt-name person  # local vision model, no key
fk dataset mask ./poses --expand 1.6             # detect, review, export masks/
fk dataset validate ./poses --require-masks      # before the run, not after it

fk serve                                         # on a lab node
fk dataset push ./poses --to olympus-2 --sidecars-only
fk fleet status

fk node models                                   # what this node can train
fk train --model flux2 --dataset poses --masked --steps 2000 --watch
```

Captioning is local by default and no image leaves the node.
**JoyCaption** is the one this was built around — a LLaVA model loaded
in-process, fine-tuned for training captions and unwilling to refuse a
subject; **Ollama** needs nothing installed into the process. Claude is
one setting away for the sets where sending images out is acceptable.
`fk node captioners --test` says whether a backend will work before a
batch does.

Prompts are saved and reusable: `fk prompts save mara-portrait "..."`,
then `--prompt-name mara-portrait` on any node. Five are shipped.

`fk --help` lists the rest. Every command is an API client, so the same
one works locally and against a node over an SSH tunnel; with no daemon
running it starts one for the length of the command rather than taking a
different code path. Everything runs headless on Windows and Linux, and
nothing under `core/` imports a UI toolkit.

## Ground rules

1. **v1 keeps working.** No changes to `AI_Image_Trainer` except bug
   fixes until this is genuinely ahead.
2. **`klein_trainer/` gets ported, not rewritten.** 4,165 lines of
   working ML code, and the only reason this project is interesting.
3. **Nothing in `core/` imports a UI toolkit.** Enforced by a test.
4. **Every dataset operation goes through `DatasetItem`.** No function
   invents its own idea of which files belong together.
5. **Anything the UI can do, the API can do.** No feature reachable only
   by clicking, or the fleet becomes second-class.
6. **No platform-specific paths, launchers, or filename assumptions.**
   The fleet is Linux; the desk is Windows.
