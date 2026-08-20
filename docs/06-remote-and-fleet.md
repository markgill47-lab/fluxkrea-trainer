# 06 — remote and fleet

## The setup this has to serve

- N Linux boxes, RTX PRO Blackwell 4000, doing the training.
- One laptop, driving them over SSH.
- Windows for development, Linux in production.

v1 cannot do this. Its logic lives inside a Qt main window, so "remote
control" means X-forwarding a desktop app over SSH — slow, fragile, and
one session per box.

## Shape

```
   laptop                              node1..nodeN  (Linux, Blackwell)
 ┌──────────────┐                    ┌───────────────────────────┐
 │ CLI  ` fk `  │ ── HTTP/SSE ────►  │  fluxkrea serve           │
 │ web client   │   (SSH tunnel)     │   ├── job queue           │
 │ fleet view   │                    │   ├── core (headless)     │
 └──────────────┘                    │   └── backends → training │
                                     └───────────────────────────┘
```

The daemon runs on each GPU node and owns everything: the dataset
folders, the job queue, the training subprocesses. Clients hold no state
worth losing — close the laptop and training continues.

## Transport

**REST for control, SSE for events.**

Events are strictly server-to-client, so Server-Sent Events fit better
than WebSockets: they survive proxies, reconnect natively with
`Last-Event-ID`, need no extra dependency, and can be read with plain
`curl` while debugging. Control actions are ordinary POSTs.

The core's event types (`Progress`, `Log`, `LossPoint`, `Finished` — see
[02](02-architecture.md#events-instead-of-callback-bundles)) serialise
directly onto the stream. One event vocabulary from the training loop to
the browser.

## API surface

Versioned under `/api/v1`. This is the full contract — the GUI is one
client of it, with no privileged path.

### Node

| | |
|---|---|
| `GET /health` | version, uptime, queue depth |
| `GET /node` | hostname, OS, python, torch, CUDA, driver, disk free |
| `GET /gpus` | per-GPU name, VRAM total/used, utilisation, temperature |

`GET /node` reporting torch/CUDA/driver matters more than it looks:
Blackwell (sm_120) needs torch 2.6+ with CUDA 12.6+, and a fleet-wide
version mismatch is otherwise invisible until a run dies.

### Datasets

| | |
|---|---|
| `GET /datasets` | registered dataset folders |
| `POST /datasets` | register a path |
| `POST /datasets/{id}/scan` | rescan, returns item count + validation report |
| `GET /datasets/{id}/items` | items with caption/mask/quality state |
| `GET /datasets/{id}/items/{stem}/image` | image bytes |
| `GET /datasets/{id}/items/{stem}/mask` | mask bytes |
| `GET /datasets/{id}/items/{stem}/preview` | redacted preview |
| `GET /datasets/{id}/items/{stem}/boxes` | detected + manual face boxes |
| `PUT /datasets/{id}/items/{stem}/boxes` | replace boxes — **the remote review pass** |
| `POST /datasets/{id}/ops/{resize,rename,augment,caption,mask}` | → task id |
| `GET /datasets/{id}/validate` | orphans, missing masks, size mismatches |
| `GET /datasets/{id}/manifest` | per-item size, mtime, digest — for sync |
| `POST /datasets/{id}/import` | tar stream in, extracted into place |
| `GET /datasets/{id}/export` | tar stream out |

The box endpoints are what let the face-mask review happen from the
laptop against a dataset sitting on a node. Without them, review is
tethered to whichever machine holds the files.

### Jobs

| | |
|---|---|
| `POST /jobs` | submit a `RunSpec`, returns job id, queued |
| `GET /jobs` | list with status |
| `GET /jobs/{id}` | status, progress, resolved config |
| `DELETE /jobs/{id}` | cancel (or dequeue if not started) |
| `POST /jobs/{id}/pause` · `/resume` | Klein supports both today |
| `PATCH /jobs/{id}/config` | live config update mid-run |
| `GET /jobs/{id}/events` | **SSE stream** |
| `GET /jobs/{id}/logs?since=` | backfill, for reconnects |
| `GET /jobs/{id}/loss` | loss series + EMA + trend |
| `GET /jobs/{id}/samples` | generated sample images |
| `GET /jobs/{id}/artifacts` | checkpoints, with download |

### Tasks

Dataset operations are long-running too, and get the same treatment:
`GET /tasks/{id}` and `GET /tasks/{id}/events`.

## Dataset placement

**Nodes have local storage, not a shared mount.** So a dataset lives on
specific nodes, and getting it there is part of the job.

### Who knows what lives where

Nobody centrally — consistent with having no coordinator. The client asks
each node `GET /datasets` and assembles the picture. `fk dataset where
poses` prints the nodes that have it, and flags any whose manifest
disagrees.

### Moving bytes

**Control goes over the API; bulk data does not.** Reimplementing rsync
over HTTP would be a poor use of effort when every node is already
reachable over SSH.

```
fk dataset push ./poses --node olympus-2
```

1. `GET /datasets/{id}/manifest` from the target.
2. Diff against local — only differing files are candidates.
3. Transfer by the best transport available:
   - **rsync over SSH** where present. Incremental, compressed, resumable.
   - **tar stream to `POST /datasets/{id}/import`** otherwise. Portable,
     works from a Windows laptop with nothing installed, one request.
4. `POST /datasets/{id}/scan` on the target to re-index.

Windows is the reason for the fallback: OpenSSH ships with Windows 10+
but rsync does not, so the tar path has to work unaided.

### The sidecar shortcut

Images are large and static. Captions and masks are small and change
constantly. Once a dataset's images are on a node, the review loop is
cheap:

```
review locally (or remotely via the box endpoints)
  → fk dataset push ./poses --node olympus-2 --sidecars-only
  → kilobytes move, not gigabytes
```

`--sidecars-only` restricts the diff to `.txt` and `masks/*.png`. This is
what makes iterating on a mask pass across a fleet practical: re-detect
with a different expansion factor, push a few hundred KB, retrain.

### Drift

Because copies are independent, they drift. `GET /datasets/{id}/manifest`
carries a digest per item so `fk fleet datasets` can show which nodes
disagree, and `validate` runs per node. Detecting drift is in scope;
automatic reconciliation is not — the client reports and the human
decides which copy wins.

## Job queue

One training job per GPU at a time; the rest queue. The queue persists to
disk so a daemon restart or a reboot does not lose submitted work, and
in-flight jobs are marked interrupted rather than silently vanishing.

Multi-GPU nodes get one queue slot per device, with the device pinned in
the `RunSpec`.

## Fleet

**Client-side aggregation. No coordinator.**

A coordinator would be a single point of failure and a second daemon to
deploy for a lab-sized fleet. Instead the client holds the node list:

```toml
# ~/.fluxkrea/fleet.toml
[[node]]
name = "olympus-1"
url  = "http://localhost:8471"   # via ssh -L
[[node]]
name = "olympus-2"
url  = "http://localhost:8472"
```

`fk fleet status` fans out and prints one table. The web client does the
same across tabs. Adding a node is a config line, and a node being down
degrades to one missing row rather than breaking the view.

Job placement is explicit — `fk train --node olympus-2` — not automatic
scheduling. Automatic placement is a later question, and a coordinator is
the price of it.

## Security

The API can launch processes and rewrite dataset folders. It is
effectively remote code execution scoped to the node.

- **Binds `127.0.0.1` by default.** Remote access is an SSH tunnel,
  which matches the existing workflow and needs no new auth story.
- **Binding to a non-loopback address requires a token.** The daemon
  refuses to start listening beyond localhost without one — no silent
  open port.
- **No secrets in config files.** The Claude API key comes from the
  environment or an OS keyring, never from a committed file. This is what
  currently forces `gui_config.json` to be gitignored, and therefore
  unshareable across the fleet.
- The API is not hardened for a hostile network and should not be exposed
  to one.

## CLI

The CLI is a full API client, not a shortcut layer. Every GUI action has
a command, because both go through the same endpoints:

```bash
fk node status
fk dataset scan  ./poses
fk dataset mask  ./poses --expand 1.6 --detector yunet
fk dataset validate ./poses
fk dataset where poses                              # which nodes have it
fk dataset push  ./poses --node olympus-2
fk dataset push  ./poses --node olympus-2 --sidecars-only
fk train --spec runs/blizzard.toml --node olympus-2
fk jobs watch <id>          # tails the SSE stream
fk fleet status
fk fleet datasets           # placement + drift across the fleet
```

This is also what makes the whole thing scriptable — a nightly sweep, a
batch of LoRAs queued across the fleet — without a GUI in the loop.

## Cross-platform rules

- `pathlib` everywhere; no backslash literals, no drive-letter
  assumptions.
- **Case-sensitive filename discipline.** v1 already carries a comment
  about this in its `dataset_Manager` import shim, which is a warning
  sign from a Windows-developed module meeting Linux.
- The daemon runs as a systemd user unit on Linux and a plain console
  process or service on Windows. No `.bat`-only entry points.
- Config lives in an OS-appropriate location, not next to the source.
- CI, when it exists, runs the core test suite on both platforms.
