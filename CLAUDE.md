# Working on FluxKrea Trainer 26

A LoRA training tool for FLUX.2 Klein and Krea 2, built as a headless
core, a per-node HTTP daemon, and clients of that daemon. Rewrite of
`D:\Projects_26\AI_Image_Trainer` (v1), which is still the working tool.

**Read [docs/00-build-handoff.md](docs/00-build-handoff.md) first.** It
holds the current state, the decisions already taken, and what is next.
Everything below is the part you need in your head while typing.

## Commands

```bash
pytest                    # 731 - core, daemon, CLI, backends
cd web && npm test        # 55  - component tests, jsdom + preact
cd web && npm run build   # the daemon serves web/dist, not web/src
.\serve.ps1               # start the daemon (Windows). ./serve.sh on Linux
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

## The rules that cost a real training run

- **Masks resample with NEAREST.** A grey pixel in a mask is a partial
  loss weight - a soft leak of the region being excluded.
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
- **That terminal must be outside the Claude desktop app.** A shell inside
  a Windows MSIX package hands every child a private `%APPDATA%` and
  `%LOCALAPPDATA%`: same path strings, different files, invisible from
  within. The daemon then reads a config nobody edits and writes one nobody
  reads. `fk serve` prints a warning when it detects this - believe it, and
  check the first four startup lines before believing anything else.

## Environment

| | |
|---|---|
| `.venv` | This package, editable. **No torch, deliberately** |
| ai-toolkit | Run with *v1's* interpreter, `AI_Image_Trainer\.venv` |
| Config | `%APPDATA%\FluxKrea\config.toml`; `fk config show` resolves it |
| Daemon | `127.0.0.1:8471` |

Shell is PowerShell 5.1: no `&&`, no ternary, and redirecting a native
command's stderr wraps each line in an `ErrorRecord` and sets `$?` false
on a clean exit. `serve.ps1` documents the three places that bites.

## Writing style

Comments and commit messages here explain *why*, and name the failure that
motivated the code, because most of this file is lessons that were paid
for. Match that. Do not add comments that restate the line below them.
