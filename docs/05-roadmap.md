# 05 — roadmap

## Honest sizing

~16,800 lines of first-party Python in v1, less ~1,000 for the dropped
Kohya stack. This is not a weekend. The plan is built so that **something
useful ships early** and v1 keeps running throughout.

Port-vs-rewrite, decided up front:

| | |
|---|---|
| **Ported near-verbatim** | `klein_trainer/` (4,165 lines), `captioners/`, `flux_presets.py`, `diagnostics.py` |
| **Rewritten** | `ui_manager.py`, `dataset_Manager/`, the ai-toolkit and Klein managers, config handling |
| **Harvested** | `utils.py` — only the toolkit-free helpers; the Qt widget code goes |
| **Dropped** | the entire Kohya / sd-scripts stack, `training_config.toml` / `dataset.toml` with it, and all of PyQt6 |

## Phases

Each phase ends somewhere usable, not mid-refactor.

**P0 — Scaffold.** `pyproject.toml`, package skeleton, `core/paths.py`,
typed config, event types, test harness, the no-UI-in-core guard test.
Windows and Linux both green.
*Done when:* `pytest` runs clean on an almost-empty tree, on both
platforms.

**P1 — Dataset core, headless.** `DatasetItem`, scanner, resize, rename,
augment, validate. Ported from v1 with the `d1890ce` fixes carried
forward rather than re-derived. CLI front end.
*Done when:* every v1 dataset operation runs from the command line with
real tests, including the mask-aware sidecar handling v1 never had.

**P2 — Face masking.** Detector protocol, YuNet, box expansion, mask
export, sidecar box persistence. Still headless.
*Done when:* a folder of pose references produces a `masks/` folder that
Krea 2 trains against, and the pose LoRA stops fighting the character
reference. **This is where the project has already paid for itself**,
before any UI exists.

**P3 — Daemon and API.** HTTP app, job queue, SSE, node and dataset
endpoints, manifest/import/export and `fk dataset push`. `fk` CLI as a
real API client. Systemd unit for the fleet.
*Done when:* a dataset is pushed to a lab box from the laptop, a dataset
op runs there over an SSH tunnel with live progress, and
`--sidecars-only` moves a re-masked pass in kilobytes.

**P4 — Backends.** `TrainingBackend` protocol, `RunSpec`, ai-toolkit
backend (FLUX and Krea 2 as one config-driven class). Job submission end
to end.
*Done when:* a Krea 2 run launches on a node from the laptop and matches
what v1 produces locally.

**P5 — Klein backend.** Port `klein_trainer/`, wrap it in the protocol.
Lift Klein's analytics — trend, outliers, EMA, export — up into
`analytics/loss.py` so both backends gain them.

**P6 — Web client.** Gallery, the mask review canvas, config, training
monitor. Served as static assets by the daemon. The review canvas is the
piece that most needs to be pleasant, and the API-driven file browsing
is the piece that most needs design attention, since there is no native
picker to fall back on.

**P7 — Fleet view.** Node list, aggregate status, job placement,
cross-node queue visibility.

**P8 — Cutover.** Install scripts for both platforms, Olympus
deployment, v1 archived.

Note P2 lands before the daemon and P3 before any UI. Both are
deliberate: the originating feature ships first, and the remote path is
proven before anything is built on top of it.

## Risks

**The GUI is where the hidden requirements live.** 2,420 lines of
`ui_manager.py` encode behaviour nobody wrote down — auto-detecting model
paths, live config updates mid-run, steps recalculation, preset-per-model
filtering. Expect to keep rediscovering these. Mitigation: read the v1
method before writing its replacement, every time; the inventory in
[doc 01](01-v1-audit.md#inventory) is the checklist.

**`klein_trainer/` must not be "improved" during the port.** It works. It
moves across, gets wrapped, and is only touched afterwards. The
temptation to clean it up while moving is the most likely way to break
training.

**Two apps, one set of models and datasets.** During P1–P7 both v1 and v2
can point at the same folders. The dataset ops are destructive (rename in
place). Do not run both against one folder at once.

**The API is remote code execution.** It launches processes and rewrites
dataset folders. Localhost-bound by default, tokens required for any
wider bind, and never exposed to an untrusted network. See
[doc 06](06-remote-and-fleet.md#security).

**Fleet version skew.** Blackwell needs torch 2.6+ / CUDA 12.6+, and a
node that has drifted will fail at runtime in confusing ways. `GET /node`
reporting torch, CUDA and driver exists to make that visible before a job
is submitted.

**Scope drift into the trainer itself.** This is a rewrite of the
*tooling* around training. It is not a new training implementation.

## Open questions

1. **Does the v1 training GUI launch from `flux_aitoolkit_manager.py`,
   or do you hand-run `configs/*.yaml`?** Determines whether the
   generated config is a real artifact or a fiction the GUI maintains,
   and where `mask_path` has to be threaded through.

2. **Klein 4B/9B vs Krea 2 — which is primary now?** v1's README says
   Klein; every recent config says Krea 2. Whichever it is gets its
   backend ported first (P4 vs P5 order).

3. **How many nodes, and are they named/reachable consistently?** The
   fleet file needs stable names and tunnel ports; worth fixing a
   convention before P3 rather than after.

4. **Web client stack.** Narrowed to four expensive-to-reverse choices —
   framework, virtualizer, chart approach, icon set. See
   [10](10-graphics-stack.md#what-to-decide-before-building).

5. **Package name.** Scaffolded as `fluxkrea` inside
   `D:\Projects_26\FluxKrea_Trainer26`. Free to change now, annoying
   later.

Resolved: Kohya dropped ([01](01-v1-audit.md#dropped-kohya--sd-scripts));
UI is a web client ([02](02-architecture.md#ui-layer-web-client));
dataset storage is node-local ([06](06-remote-and-fleet.md#dataset-placement)).
