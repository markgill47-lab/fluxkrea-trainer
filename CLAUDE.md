# Working on FluxKrea Trainer 26

A LoRA training tool for FLUX.2 Klein and Krea 2, built as a headless
core, a per-node HTTP daemon, and clients of that daemon. Rewrite of
`D:\Projects_26\AI_Image_Trainer` (v1), which is still the working tool.

**Read [docs/00-build-handoff.md](docs/00-build-handoff.md) first.** It
holds the current state, the decisions already taken, and what is next.
Everything below is the part you need in your head while typing.

## Commands

```bash
pytest                    # 817 - core, daemon, CLI, backends
cd web && npm test        # 69  - component tests, jsdom + preact
cd web && npm run build   # the daemon serves web/dist, not web/src
.\serve.ps1               # start the daemon (Windows). ./serve.sh on Linux
fk serve --lab            # bind the LAN for a room of students. See deploy/
```

`FLUXKREA_AITOOLKIT` points `tests/backends/test_against_real_aitoolkit.py`
at a real ai-toolkit checkout; use `ai-toolkit-krea2`, not the rotted
`ai-toolkit` beside it.

## The rules a test enforces

1. **Nothing under `core/` imports a UI toolkit, an HTTP framework, an
   HTTP client (`requests`, `httpx`), or `fluxkrea.daemon` / `fluxkrea.cli`.**
   Captioners talk to Ollama through stdlib `urllib` for exactly this
   reason.
2. **Every dataset operation goes through `DatasetItem`.** One place knows
   that a bundle is image + caption + mask. Three hand-rolled copies of
   that in v1 is why this project exists.
3. **Anything the UI can do, the API can do.** No feature reachable only
   by clicking, or the fleet becomes second-class.
4. **No platform-specific paths or launchers.** The fleet is Linux; the
   desk is Windows. Both suites run on both.

## The shapes of the thing

- **A project is a grouping, not an owner.** It holds a name, a list of
  dataset *ids* and a shared training config. Datasets stay in the node's
  registry exactly as they were, so `fk dataset register` still works and
  one folder can be in two projects. Deleting a project deletes a
  grouping and nothing on disk - same promise `forget` makes.
- **A project id never moves.** Renaming changes the label only. The id is
  what every open browser and every queued job holds; deriving a new one
  from a new name would orphan all of them silently.
- **The project is the only identity.** No accounts, no passwords. It is
  what the shared queue lists beside a run, and what the queue interleaves
  on so one student's batch cannot hold the card all day. It is a name a
  browser chose - fairness between good-faith parties, not a security
  control.
- **`daemon.lab_mode` is the one setting that opens a port.** It is not
  writable over the API, it refuses to start without `dataset.roots`, and
  the daemon prints every address it is reachable at before uvicorn says
  anything.
- **A region is a bounding box plus a shape.** `rect` or `ellipse`, and
  detection produces ellipses because a face is one. Expansion, clamping,
  hit-testing and the handles are the same arithmetic for both; only the
  fill differs. An unknown shape reads as `rect` rather than raising -
  losing a reviewer's box to a spelling is the worse fault.

## The rules that cost a real training run

- **Masks resample with NEAREST.** A grey pixel in a mask is a partial
  loss weight - a soft leak of the region being excluded.
- **An ellipse is clipped, never clamped.** Clamping clamps its bounding
  box, which moves the centre and squashes the axes - so a face at the
  edge of the frame would be covered by a different ellipse from the one
  drawn in review. `render_mask` and the Viewport canvas both let the
  drawing surface clip instead, and they must keep agreeing.
- **A LoRA's publish folder comes off the model record**, never from a
  substring of a name. A Krea 2 LoRA in `models/loras/flux2` loads,
  generates noise, and reads as a bad training run rather than a misfiled
  file. Same lesson as v1's `detect_backend`, one layer further out.
- **A run's name is derived in one place**, `core/backends/spec.py::run_name`,
  and never contains a path. Two derivations disagreed once and produced a
  264-character path, which Windows reports as a missing file.
- **Nothing written into a run folder may start with the run name.**
  ai-toolkit globs `{job_name}*` to decide whether to resume, and
  `torch.load`s the newest match. Hence `_fluxkrea.yaml`.
- **A daemon restart is required after any Python change.** Python loads a
  module once; a running daemon keeps the old one. The client shows a
  staleness banner, and it is telling the truth.
- **Do not tie the daemon's lifetime to your session.** Training runs take
  hours. Tell the user to run `.\serve.ps1` in their own terminal rather
  than starting it from a tool call.
- **`%APPDATA%` on this desk is not reliable.** A shell running inside a
  Windows app package hands its children a private view of `%APPDATA%` and
  `%LOCALAPPDATA%` - same path strings, different files, no signal from
  inside. The daemon read a `config.toml` nobody was editing and wrote one
  nobody could read, for an afternoon. `.fluxkrea/` beside `serve.ps1` is
  the answer: the scripts set `FLUXKREA_HOME` to it, so config, data, cache
  and state all live on D: where nothing can redirect them. **Check the
  first four startup lines before believing anything else** - they name the
  config file, the home, the package directory and the backend.

## Environment

| | |
|---|---|
| `.venv` | This package, editable. **No torch, deliberately** |
| ai-toolkit | Run with *v1's* interpreter, `AI_Image_Trainer\.venv` |
| Config | `%APPDATA%\FluxKrea\config.toml`; `fk config show` resolves it |
| Daemon | `127.0.0.1:8471`. `fk serve --lab` binds the LAN instead |
| Projects | `data_dir/projects.json`, beside the dataset registry |

Shell is PowerShell 5.1: no `&&`, no ternary, and redirecting a native
command's stderr wraps each line in an `ErrorRecord` and sets `$?` false
on a clean exit. `serve.ps1` documents the three places that bites.

## Writing style

Comments and commit messages here explain *why*, and name the failure that
motivated the code, because most of this file is lessons that were paid
for. Match that. Do not add comments that restate the line below them.
