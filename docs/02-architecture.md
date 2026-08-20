# 02 — architecture

## The one rule

**`core/` never imports a UI toolkit.** Everything the application
actually *does* lives in `core/` as plain Python: scanning folders,
resizing, renaming, detecting faces, writing masks, generating backend
configs, launching and monitoring training. Every interface — CLI, HTTP
API, whatever the GUI turns out to be — is a client of it.

Enforced by a test that walks `core/` and fails on any UI import.

This is what makes the fleet workflow possible at all. A core that runs
headless can be wrapped in a daemon and driven over SSH; a core that
constructs `QMessageBox` cannot.

## Three layers

```
core/       pure logic, no I/O beyond files, no UI, fully testable
  ↑
daemon/     HTTP API + job queue, one per GPU node   → see doc 06
  ↑
clients/    CLI, web or desktop UI, fleet view
```

Clients never import `core` directly, even when running on the same
machine — they go through the API. One code path, so the local case can
never quietly diverge from the remote one.

## Layout

```
FluxKrea_Trainer26/
  pyproject.toml
  src/fluxkrea/
    core/
      paths.py            # every path the app knows, resolved once
      config.py           # one typed config, one file
      events.py           # event types + emitter, no UI, no HTTP
      dataset/
        item.py           # DatasetItem — the invariant. See doc 03.
        scan.py           # folder -> list[DatasetItem]
        ops/
          resize.py       # was fix_images
          rename.py       # was mass_rename_images
          augment.py      # was create_duplicates
          mask.py         # new: face masking. See doc 04.
      detect/
        base.py           # Detector protocol
        yunet.py          # OpenCV YuNet
      captioners/         # ported from v1 as-is
      backends/
        base.py           # TrainingBackend protocol
        aitoolkit.py      # FLUX + Krea2, one class, config-driven
        klein.py          # wraps trainer/
      analytics/
        loss.py           # EMA, trend, outliers — lifted out of Klein
      trainer/            # klein_trainer/, ported near-verbatim
    daemon/
      app.py              # HTTP app
      routes/             # node, datasets, jobs, tasks
      queue.py            # persistent per-GPU job queue
      stream.py           # SSE
    cli/
      __main__.py         # `fk` — a full API client
  web/                    # browser client, served as static assets
    gallery/              #   dataset browsing
    review/               #   the mask review canvas
    monitor/              #   training progress, loss, samples
    fleet/                #   multi-node view
  tests/
```

## Events instead of callback bundles

v1 threads four callbacks through every manager constructor. v2 has
typed events:

```python
@dataclass(frozen=True)
class Progress:
    step: int
    total: int
    message: str

@dataclass(frozen=True)
class Log:
    line: str
    level: str = "info"

@dataclass(frozen=True)
class LossPoint:
    step: int
    value: float
    image_id: str | None = None

@dataclass(frozen=True)
class Finished:
    ok: bool
    detail: str = ""
```

Core operations take an `emit: Callable[[Event], None]`, defaulting to a
no-op. The CLI passes a printer. The daemon passes something that fans
out to SSE subscribers. Core never knows which.

One vocabulary from the training loop to the browser — the same dataclass
that a backend emits is what arrives on the laptop.

Cancellation is a `threading.Event` passed in and checked at loop tops,
not `progress.wasCanceled()` read off a dialog.

## The backend protocol

The thing v1 has three unwritten copies of, written down once:

```python
class TrainingBackend(Protocol):
    name: str
    def supports(self, model_id: str) -> bool: ...
    def generate_config(self, run: RunSpec) -> Path: ...
    def start(self, config_path: Path, emit: Emitter,
              cancel: threading.Event) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def progress(self) -> BackendProgress: ...
```

Two backends survive the cut: **ai-toolkit** (FLUX and Krea 2 collapse
into one config-driven class, since v1's Krea2 manager is already just a
subclass) and **Klein** (wrapping the ported trainer). Kohya is dropped —
see [01](01-v1-audit.md#dropped-kohya--sd-scripts).

Model dispatch is explicit. v1's `detect_backend` falls through to
`return 'kohya'` for anything unrecognised, so an unknown model silently
routes to the wrong trainer. v2 raises.

Notably absent from the protocol: `get_loss_history`, trend detection,
outliers, EMA. Those are **not** backend concerns. Backends emit
`LossPoint` events; `analytics/loss.py` consumes the stream and computes
the rest for every backend equally. That is the fix for Klein having five
features the ai-toolkit backends silently lack.

`RunSpec` is a typed description of a training run — model, dataset,
network dims, schedule, sampling, target device — that each backend
renders into its own config format. One source of truth, N renderers,
instead of v1's independent `generate_config` methods drifting apart.
It is also the payload of `POST /jobs`.

## Threading

Core operations are synchronous and blocking, and say so. The daemon runs
them on worker threads and streams their events. No `processEvents()` in
a loop anywhere. Training subprocesses are owned by the backend and
monitored on their own thread, as they already are in v1 — that part
works.

## Config

One file, one schema, dataclass-backed, with explicit precedence:

```
defaults in code  <  config file  <  environment  <  CLI flags
```

Secrets move out of the config file entirely (environment or OS keyring),
so the config becomes committable and shareable across the fleet — which
`gui_config.json` being gitignored currently prevents.

Backend-specific configs stay as *generated artifacts*, written from
`RunSpec`, never hand-edited as the source of truth.

## UI layer: web client

**Decided: a browser client, served by the daemon.** Qt is dropped
entirely — v1's PyQt6 is not carried forward.

The deciding factors, given the fleet:

- Nothing to install on any machine you drive from. The client arrives
  with the daemon, reached through the same SSH tunnel as the CLI.
- The mask review canvas is ordinary `<canvas>` work rather than custom
  `QPainter` painting — and that screen is the one that decides whether
  reviewing a few hundred images is bearable.
- Design work is directly implementable rather than a reference sketch.

What it costs, and how it is handled:

- **A second stack.** Kept small: static assets served by the daemon, no
  server-side rendering, no coupling between a build step and the API.
  The client is just another API consumer, exactly like `fk`.
- **No native file dialogs.** Paths come from the API — dataset and
  folder browsing are endpoints (`GET /fs/browse`, scoped to configured
  roots), not an OS picker. This is a genuine ergonomic loss on the
  desktop and the main thing the design work needs to solve well.

Note that X-forwarding was never the issue: a Qt client on the laptop
talking HTTP to a remote node would have worked fine. The decision rests
on install friction, canvas work, and design reuse.

Both options sit on the same core and the same API, so this decision
changes the client layer only — but it should be made before any design
work, because it determines whether that work is directly usable.

## Testing

- `core/` gets real unit tests against temp folders — the dataset ops in
  particular, which is where the v1 bugs lived.
- Guard test: no UI imports under `core/`.
- One integration test per backend that generates a config and asserts
  its shape, without launching training.
- API contract tests against a daemon with a fake backend.
- Mask round-trip: detect → write → read back → confirm polarity, size
  and alignment against the source image.
- Core suite runs on Windows and Linux both.
