# 01 — v1 audit

Survey of `D:\Projects_26\AI_Image_Trainer` as of 2026-08-20, at commit
`d1890ce`. ~16,800 lines of first-party Python, excluding the vendored
`ai-toolkit`, `kohya_ss`, `sd-scripts` and `bfl_flux2` checkouts.

## Inventory

| Module | Lines | Verdict |
|---|---:|---|
| `ui_manager.py` | 2,420 | **Rewrite.** God object, 66 methods, one class. |
| `klein_trainer/` | 4,165 | **Port intact.** The real asset. |
| `dataset_Manager/` | ~1,600 | **Rewrite** onto the new dataset model. |
| `utils.py` | 626 | **Mostly dies with Qt.** Only the toolkit-free helpers survive — `format_time`, `format_bytes`, path validation. |
| `flux_aitoolkit_manager.py` | 493 | **Rewrite** as a backend implementation. |
| `klein_gui_manager.py` | 469 | **Fold** into the Klein backend + analytics. |
| `flux_presets.py` | 465 | **Port.** Data, mostly. |
| `kohya_*.py` (4 files) | ~1,000 | **Dropped.** See below. |
| `enhanced_training_manager.py` | 347 | **Replace** with the backend registry. |
| `gui_config_manager.py` | 319 | **Replace** with the typed config. |
| `krea2_aitoolkit_manager.py` | 231 | **Fold** into the ai-toolkit backend. |
| `captioners/` | ~400 | **Port as-is.** Already clean. |
| `diagnostics.py` | 197 | **Port.** Crash handlers, still useful. |
| `scripts/` | 13 files | **Triage.** Mostly one-off probes. |

## Dropped: Kohya / sd-scripts

Roughly 1,000 lines and an entire backend, removed from scope before a
line of v2 is written. The evidence it is dead:

- `enhanced_training_manager.detect_backend` routes to `kohya` only for
  `stable-diffusion`, `sdxl` or `stabilityai` model names. This project
  trains FLUX.2 Klein and Krea 2.
- Every config in `configs/` is ai-toolkit or Krea2.
- `kohya_config_manager.py`, `kohya_dataset_manager.py`,
  `kohya_training_manager.py` and `kohya_presets.py` were last touched
  2026-01-13. Everything else has changed through July and August.

One behaviour worth carrying forward as a *fix* rather than a port:
`detect_backend` ends with `return 'kohya'` as the fallback for unknown
models, so an unrecognised model silently routes to a trainer that cannot
handle it. v2 raises instead.

## The backend problem

Three stacks implement the same lifecycle (four, before dropping Kohya):

1. **ai-toolkit FLUX** — `flux_aitoolkit_manager.AIToolkitManager`
2. **ai-toolkit Krea2** — `krea2_aitoolkit_manager.Krea2AIToolkitManager`,
   subclassing the FLUX one
3. **Klein** — `klein_gui_manager.KleinGUIManager` over `klein_trainer/`

with `EnhancedTrainingManager.detect_backend()` dispatching between them.

Each independently defines `start_training`, `stop_training`,
`is_training_running`, `get_progress` and `get_loss_history`. So an
interface already exists — it is just never written down, never type
checked, and never uniformly honoured.

The features are badly asymmetric. Klein alone has trend detection,
outlier images, EMA series, metric export and live config updates
(`klein_gui_manager.py:344-467`). The ai-toolkit backends have
`get_progress` and `get_loss_history` and nothing else. Anything built on
the richer API silently degrades on the other backend.

**v2 fix:** one declared `TrainingBackend` protocol, with all analytics
lifted *above* the backend line so every backend gets them for free. See
[02 — architecture](02-architecture.md#the-backend-protocol).

## Qt welded to logic

`ImageProcessor` imports `QApplication`, `QProgressDialog` and
`QMessageBox` and constructs dialogs inside its processing loops.
Consequences:

- Nothing can be unit tested without a Qt application object.
- **Nothing can run headless**, from a script, or over SSH — which makes
  the whole fleet workflow impossible. This is the single most expensive
  structural mistake in v1.
- Progress is reported by `QApplication.processEvents()` on the main
  thread, so every long operation half-freezes the UI.

`utils.py:301` already has a `WorkerThread(QThread)` that nothing in the
processing path uses.

## Callback injection as an ad-hoc event bus

Every manager constructor takes the same bundle:

```python
def __init__(self, update_status_callback, update_training_output_callback,
             show_error_callback, show_info_callback, ...):
```

An event system with no registry, no typing, and no way for two listeners
to observe the same event — which is exactly what a daemon streaming to
multiple remote clients needs. Replaced in v2 by typed events.

## Config sprawl

Four overlapping stores with no documented precedence:

| File | Written by | Holds |
|---|---|---|
| `gui_config.json` | `GUIConfigManager`, dotted key paths | UI state, API keys, captioner settings |
| `training_config.toml` | `KohyaConfigManager` | Kohya training args (dying with Kohya) |
| `dataset.toml` | `KohyaDatasetManager` | Kohya dataset definition (likewise) |
| `configs/*.yaml` | `AIToolkitManager.generate_config` | Generated ai-toolkit job configs |

`gui_config.json` and `flux_gui_config.json` are both gitignored for
holding API keys and local paths — meaning the app's actual configuration
is untracked and cannot be shared between the Windows dev box and the
Linux fleet. Dropping Kohya removes two of the four stores outright.

## Dataset-side rot (fixed in `d1890ce`)

Found and fixed in v1 before starting this plan, listed here because v2
must not reintroduce them:

- EXIF orientation silently discarded on resize, rotating photos 90°.
- PIL file handles never released, blocking later renames on Windows.
- `mass_rename_images` leaving folders half-renamed with no rollback.
- Selection renames always restarting numbering at 1 and aborting.
- `mass_rename_images` never returning the `success` key its own caller
  branches on, so successful renames reported "Error: Unknown error".

Still open, deliberately deferred to v2:

- Two independent copies of the supported-image-extension list
  (`ImageProcessor.get_image_files` and `DataManager.get_image_files`).
- Captions stored in *both* `.txt` sidecars and a JSON file, with
  precedence resolved ad hoc in `DataManager._get_description_for_image`.
- `dataset_Manager/image_gallery_simple.py` is a dead entry point using
  absolute imports that only work if the cwd happens to be inside the
  package.

## What is genuinely good

Not everything needs replacing:

- **`klein_trainer/`** — a real trainer: `model.py`, `lora.py`, `vae.py`,
  `dataset.py`, `trainer.py`, `analytics.py`. Ports across.
- **`captioners/`** — clean `Captioner` base with Claude, Ollama and
  JoyCaption backends behind a factory. Exactly the shape everything else
  should have had.
- **`flux_presets.py`** — mostly data, keyed by model.
- **`diagnostics.py`** — crash handlers and resource checks.

## Deployment reality

Per `README.md` and `install_olympus.sh`, v1 already runs on **Windows for
development and Linux (Ubuntu 22.04+) on the Olympus lab machines**,
installed by a curl-to-bash script that clones ai-toolkit and applies
local patches. The Blackwell cards need torch 2.6+ with CUDA 12.6+
(sm_120).

What v1 does *not* support is driving those machines remotely, which is
the actual working pattern. See [06 — remote and fleet](06-remote-and-fleet.md).
