# 00 — build handoff

Read this first if you are starting a build session. Everything before
this point was planning; this is the state it left behind and the first
thing to do.

## Where things stand

**P0 through P6 are built and green.** 623 tests pass on Windows; the
suite is platform-neutral and both OS layouts are exercised from either
one. What exists:

| | |
|---|---|
| `core/paths.py` | Every location, OS-appropriate, all env-overridable |
| `core/config.py` | One typed config, `defaults < file < env < flags`, secrets refused in the file |
| `core/events.py` | `Progress` / `Log` / `LossPoint` / `Finished`, emitter combinators, cancellation |
| `core/imaging.py` | EXIF, handle release, JPEG settings, the NEAREST mask rule |
| `core/dataset/` | `DatasetItem`, one scanner, metadata cache, `validate` |
| `core/dataset/ops/` | `resize`, `rename` (plan/execute, rollback), `augment`, `mask`, `caption` |
| `core/captioners/` | `Captioner` interface; JoyCaption in-process, Ollama, Claude, behind a registry |
| `core/captioners/prompts.py` | Saved caption prompts, five shipped, built-ins shadowed not destroyed |
| `core/backends/plan.py` | `images x repeats x epochs`, and a duration measured from this node's own runs |
| `core/analytics/loss.py` | EMA, trend, outliers, LTTB decimation - above the backend line |
| `core/detect/` | `Detector` protocol, YuNet, null detector, registry |
| `core/dataset/manifest.py` | Per-file size, mtime, digest; the diff sync rests on |
| `core/dataset/archive.py` | Tar streaming, with extraction that refuses to escape |
| `core/backends/` | `TrainingBackend` protocol, `RunSpec`, model registry |
| `core/backends/aitoolkit.py` | **FLUX.2, Klein and Krea 2 in one config-driven class** |
| `daemon/` | 30 endpoints, task runner, SSE, persistent job queue, token and path scoping |
| `cli/` | `fk` — a real API client: dataset ops, push, fleet, jobs, `serve` |
| `deploy/` | systemd user unit and deployment notes |

The guard test is in place and passing: nothing under `core/` imports a UI
toolkit, an HTTP framework, or a client package — and it now resolves
relative imports, which is how a `core → daemon` reference slipped past it
once already.

The web client is built: dataset gallery, mask review, training monitor
and settings, in `web/` (Vite + Preact + TypeScript, self-hosted fonts,
no CDN). The daemon serves the built client, so there is one thing to
deploy per node.

**Not built:** the Klein backend (P5 — the standalone `klein_trainer/`,
not Klein-through-ai-toolkit, which works today). The fleet view is not
being built as a node-served UI at all — see the decision below.

### FLUX.2 works

`fk train --model flux2 --dataset poses --masked` renders a config,
launches ai-toolkit, and streams progress and loss back. Four models go
through the one backend: `flux2`, `flux2-klein-4b`, `flux2-klein-9b` and
`krea2`, plus `flux1` for the older stack.

Verified against the real ai-toolkit rather than against our own
expectations: `tests/backends/test_against_real_aitoolkit.py` hands each
generated config to ai-toolkit's own `get_job` and asserts it resolves to
an `SDTrainer` with the right architecture. Those tests skip when no
checkout is present; point `FLUXKREA_AITOOLKIT` at one to run them.

**A v1 bug found on the way.** v1 emits `arch: flux2_klein`, which is not
an architecture ai-toolkit registers — the real ones are per size. Its
Klein configs would fail at load. Confirmed by enumerating the arch
strings in `extensions_built_in/diffusion_models/`.

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
| Fleet **UI** | Not in the node-served client. See below |
| Dataset storage | Node-local; manifest diff over rsync or tar |
| UI | Browser client served by the daemon. **PyQt6 is not carried forward** |
| Face detection | OpenCV YuNet behind a pluggable `Detector` interface |
| Mask delivery | ai-toolkit `mask_path` loss masking, not baked-in pixels |

## Open — decide before the phase that needs it

The four client-stack choices from
[10](10-graphics-stack.md#what-to-decide-before-building) are made:
Preact, TanStack Virtual, uPlot, Lucide. The log viewer deviates from
doc 10 in one respect — rows are one line each with horizontal scroll
rather than wrapping, because variable-height measurement of wrapped
rows drew them on top of each other.

Two product questions remain in [05](05-roadmap.md#open-questions): how
many nodes there are and whether they are named consistently, and the
package name. **Resolved: FLUX.2 first** — P4 shipped as the ai-toolkit
backend, which covers FLUX.2, both Klein sizes and Krea 2 together.

One design leftover: the state-surface tints are not normalised. Success,
running, paused and several amber depths are still floating as literals
in the screen compositions rather than tokens. Not blocking; fix before
P6.

## Start here: a real run, then P5

**Point it at a real checkout and train something.** Everything below the
GPU is proved; nothing above it is.

```bash
export FLUXKREA_BACKENDS_AITOOLKIT_PATH=/path/to/ai-toolkit-krea2
export FLUXKREA_BACKENDS_PYTHON_EXE=/path/to/ai-toolkit/.venv/bin/python
fk node models                     # the backend should read "ready"
fk train --model flux2 --dataset poses --masked --steps 100 --watch
```

What a first real run will find, and no test here can: whether the step
and loss regexes match this build's actual output, how long the Mistral
text encoder takes to load, and whether `low_vram` is set right for the
card. The regexes are in `core/backends/aitoolkit.py`; `OutputParser` is
tested standalone, so a mismatch is a one-line fix plus a test.

Then **P5 — the Klein backend**: porting the standalone `klein_trainer/`
(4,165 lines) and wrapping it in the protocol. Note that Klein *through
ai-toolkit* already works, so P5 is only worth the effort if the
standalone trainer does something the ai-toolkit path does not.

Either way, `analytics/loss.py` is the piece that pays for itself: lift
Klein's trend detection, outliers and EMA above the backend line so every
backend gets them from the `LossPoint` stream. The queue already keeps the
series (`GET /jobs/{id}/loss`); nothing derives from it yet.

## Decisions taken during the build

Places where the spec left room and the code had to pick:

- **Resize refuses to enlarge by default.** v1 conflates "smaller than the
  target" with "corrupt" via a hard 512 floor in its validator. v2 reports
  those images as `too_small` — not a failure — and `--upscale` opts in.
- **Ops emit `Progress` and `Log`, never `Finished`.** The runner owning
  the operation emits exactly one terminal event, so composing operations
  cannot produce two.
- **A corrupt `face_boxes.json` raises; a corrupt `metadata.json` does
  not.** The first holds human review work, the second is a cache.
- **The CLI never calls `core`, even with no daemon running.** Doc 02
  forbids a divergent local path, and the usual "local mode" is exactly
  that divergence. Instead `fk` starts the real daemon on an ephemeral
  loopback port for the life of one command — same routes, same socket,
  about a second of startup. `fk serve` avoids the startup.
- **`RunSpec` lives in `core/backends/`, not in the daemon.** The backend
  protocol takes one, and core may not reach up into its clients. The
  daemon imports it.
- **A `--dry-run` rename is the same endpoint.** The operation returns the
  plan instead of executing it, rather than growing a second route that
  could drift from the first.
- **Resize enlarges by default.** A bucket of mixed resolutions is worse
  for training than a few upscaled images. `--no-upscale` restores the
  refusal, and those images are reported as `too_small` rather than
  counted as errors.
- **Models are named, never inferred.** v1 reads `flux`, `klein` and `4b`
  out of the checkpoint path, so a file named for one model trains as
  another. `core/backends/models.py` names each one with the settings that
  follow from it.
- **The node-served client has no fleet view.** The client is served by
  each node, so a fleet tab needs the node list from somewhere: either the
  daemon serves it and every node then knows about every other, or the
  browser reaches each node directly. Both break "client-side aggregation,
  no coordinator", and since the API is remote code execution scoped to a
  node, chaining it means one compromised UI reaches all of them. Fleet
  aggregation stays where the node list already lives — the operator's own
  machine, via `fk fleet status`. `fleet.toml` is read only by
  `cli/fleet.py`; the daemon has no fleet awareness at all, and that is
  worth keeping.
- **`PUT /config` will not write `daemon.*` or `dataset.roots`.** Every
  other setting is editable from the settings screen, because a caller who
  can already launch training processes is not meaningfully restrained by
  a mask feather. Those two are different: `dataset.roots` is the
  allow-list every path check is measured against, and `daemon.host` is
  what keeps the API on loopback. Widening either widens the API's reach
  rather than changing a preference, so they are edited on the node by
  someone with a shell on it. `GET /config` returns `read_only` so the UI
  can say why rather than discovering it through a 403.
- **A secret-looking config key is only a secret if its value is a
  string.** The name test alone is a substring match, and `max_tokens`
  contains "token". A number named after a secret is a count; refusing to
  load a config over one would be a bug wearing a security hat.
- **A duration estimate is measured or absent.** A seconds-per-step
  constant is wrong on every card it was not measured on, and wrong
  differently at each resolution and rank. `plan.py` takes the rate from
  runs that already finished on this node, reports which runs it used, and
  shows nothing at all when there are none - because somebody plans an
  evening around that number. Runs under 50 steps are ignored: they
  describe model load and caching, not the per-step rate.
- **A run's name is derived in exactly one place, and never contains a
  path.** It was derived in two: the daemon used the dataset's basename for
  the output folder, the backend slugged the dataset's whole absolute path
  for the config. They disagreed, so the config was written outside the run
  it described - and on Windows the doubled path reached 264 characters,
  past MAX_PATH, which surfaces as `FileNotFoundError` on a file the code
  had just tried to create. `core/backends/spec.py::run_name` is now the
  only derivation, and `config_path` sits inside the run's own output.
- **ai-toolkit appends the job name to `training_folder` itself**
  (`BaseTrainProcess.py:45`: `save_root = join(training_folder, name)`), so
  handing it the run's own folder wrote checkpoints and samples to
  `runs/<name>/<name>/` - one level below everything that looks for them,
  including the monitor's sample strip. It gets the *parent* now, and
  `AIToolkitBackend.output_folder()` is the one answer to "where does this
  run write". The fake trainer in `tests/` wrote samples where they were
  expected rather than where ai-toolkit puts them, which is exactly why
  that test hid the bug - `test_plan.py` now pins our folder against
  ai-toolkit's own formula.
- **A failed config write says why.** Windows reports a too-long path as a
  missing file, which sends you looking for the wrong thing. The write is
  wrapped and re-raised with the length, the path, and the three ways out.
- **A screen that can start an expensive job does not own the state that
  decides what it starts.** The training form kept its settings locally,
  so switching to the monitor and back unmounted it and silently reset
  every field - the dataset falling back to the first registered one. A
  run then trained the wrong images and said nothing, which cost a real
  run. The form's state lives in the screen above it and in
  sessionStorage, and the dataset is app-wide rather than a second
  selection that can drift from the one in the top bar.
- **A screen that can start an expensive job restates what it is about to
  do.** The dataset id, its path and its image count sit directly above
  the button. A typed run name is not evidence of the dataset that was
  picked.
- **Training settings lock while a run is going, rather than hiding.** The
  settings a run is using are worth reading while it runs. A field that
  accepts an edit which changes nothing is the thing to avoid.
- **JoyCaption is a HuggingFace model, not an Ollama one.** It never
  appears in `ollama list` and the weights live in `~/.cache/huggingface`.
  Confusing the two costs an afternoon looking for a 16GB model that is
  sitting right there; the module docstring says so.
- **A readiness probe checks the config and the weights, not the whole
  repo.** `snapshot_download(local_files_only=True)` reports a working
  model as "not cached, will download 16GB" when `.gitattributes` and the
  licence are missing. It said exactly that about a model that had
  captioned an image minutes earlier.
- **A prompt that lists what to cover must also say how to write.** Asked
  to cover "pose, expression, framing, clothing", JoyCaption sometimes
  reads the list as a form and answers `**Pose:** Standing.` Two captions
  in forty-two on the Mara set. The shipped prompts now demand prose, and
  `_clean` strips markdown emphasis regardless — a LoRA trained on
  asterisks learns asterisks.
- **The captioner is closed when a batch ends.** JoyCaption holds 9-17GB
  on the same card that runs training.
- **The captioner is probed once per batch, before the loop.** A stopped
  Ollama daemon is one message, not two hundred. Five failures in a row
  abort the run, because past a handful it is the backend that is broken
  rather than the images.
- **A masked run validates before it launches.** `require_masks` runs as a
  preflight, because ai-toolkit trains an image with no matching mask
  unmasked and says nothing.

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

## The against-real-ai-toolkit tests

`tests/backends/test_against_real_aitoolkit.py` hands each generated
config to ai-toolkit's own `get_job()` and asserts it resolves to a
trainer with the right architecture. It finds a checkout through
`FLUXKREA_AITOOLKIT`, falling back to the two under
`AI_Image_Trainer/`, and runs it in **v1's interpreter** - ours has no
torch, deliberately.

Point it at **`ai-toolkit-krea2`**. All 7 pass there. The older
`ai-toolkit` beside it cannot import at all in that venv
(`cannot import name 'Repository' from 'huggingface_hub'` - removed in
hub 1.0), which is a rotted checkout rather than anything about our
configs.

## Testing the client

`cd web && npm test` — Vitest, jsdom, `@testing-library/preact`. Separate
from `pytest` on purpose: they need a DOM and a component tree, and folding
them into the Python run would make every Python test wait on npm.

What belongs there is behaviour that **only exists once a component is
mounted**: state that has to survive an unmount, a value that must not be
written when it did not change, the exact body handed to the API. Styling,
layout and "does it render" are what the browser pass is for, and neither
replaces the other.

Two notes for whoever writes the next one:

* `tests/setup.ts` stubs `matchMedia`, `ResizeObserver`, `EventSource` and
  a canvas context at **module scope**, not in a hook. uPlot reads
  `matchMedia` while being imported, so a stub installed in `beforeEach`
  arrives after the module graph has already thrown.
* A screen that loads several endpoints re-renders as they land, and a
  keystroke delivered inside that window is discarded — the parent render
  lands on top of the field's queued state. Real users cannot type that
  fast; tests can, every time. `settings.test.tsx` has a `ready()` helper
  that waits the screen out.

## Design artifacts

| File | Status |
|---|---|
| `design/tokens.css` | **Authoritative for token values.** 119 lines, drop-in |
| `design/FluxKrea Design Catalog.dc.html` | 56 component specimens + props reference |
| `design/FluxKrea Screens.dc.html` | Four screens, 1280 variant, edge states |
| `design/brief-02-screens.md` | The brief that produced the screens |
| `assets/models/face_detection_yunet_2023mar.onnx` | Vendored, 227KB. `fk node status` shows the detector ready |

The `.dc.html` files need `support.js` beside them; serve the folder over
HTTP rather than opening from disk.

Two notes on the design output. The catalog HTML loads fonts from the
Google Fonts CDN — the client must self-host, and `tokens.css` carries
the `@font-face` shape needed. And the props reference tables in the
catalog are an implementation contract worth reading before writing any
component; they encode the rules, not just the shapes.
