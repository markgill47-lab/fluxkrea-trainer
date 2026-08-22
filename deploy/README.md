# Deployment

## A fleet node (Linux)

```bash
pip install -e ".[daemon]"
cd web && npm ci && npm run build && cd ..     # see below
cp deploy/fluxkrea.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fluxkrea
systemctl --user status fluxkrea
```

**The browser client is built, not committed.** `web/dist/` is in
`.gitignore`, so a fresh clone serves JSON at `/` and says
`"client": "not built - run npm run build in web/"`. Building it needs
Node and roughly 200MB of npm downloads. A node with no network for that
can have `web/dist/` copied in from anywhere — the daemon only reads it,
and nothing else in the tree depends on it.

### Making it able to train

Installing the daemon gets you dataset tooling. Training needs four more
settings, and `fk node models` tells you when they are right - look for
`backend aitoolkit: ready`.

```bash
fk config set   backends.aitoolkit_path=/opt/ai-toolkit   backends.python_exe=/opt/ai-toolkit/.venv/bin/python   backends.output_root=/data/Output   backends.comfyui_path=/opt/ComfyUI          # optional
```

**`python_exe` is ai-toolkit's interpreter, not this package's.** This
package deliberately has no torch: the laptop driving the fleet does not
need one, and a node runs the trainer as a subprocess in the environment
that has ai-toolkit's dependencies. Pointing this at our own interpreter
gets an `ImportError` on the first run, not at configuration time.

**`output_root` is where checkpoints land**, so it wants real disk and a
backup story. Each run writes one folder:

```
Output/<run>/_fluxkrea.yaml            our generated config
Output/<run>/config.yaml               ai-toolkit's own copy
Output/<run>/<run>_000000400.safetensors
Output/<run>/optimizer.pt
Output/<run>/samples/
```

Nothing else may be put in that folder with a name starting with the run
name: ai-toolkit decides whether to resume by globbing `{run}*` there and
calling `torch.load` on the newest match. That is why our config is
`_fluxkrea.yaml` and not `<run>.yaml`.

**`comfyui_path` is optional and saves a download.** A node with weights
already in a ComfyUI `models/diffusion_models` folder uses them instead of
fetching a second copy, preferring the full-precision file over an fp8
one. Without it, and without a `backends.model_paths` entry, a model is
pulled from its HuggingFace repo on first use. Krea 2 has no public repo,
so it must be named explicitly:

```bash
fk config set backends.model_paths.krea2=/models/krea2_raw.safetensors
```

### Captioning on a node

Optional, and none of it is installed by default:

```bash
pip install -e ".[joycaption]"   # in-process LLaVA; ~16GB of weights
pip install -e ".[claude]"       # the Anthropic SDK
```

Ollama needs nothing installed here - it is a URL. `fk node captioners
--test` probes whichever is configured and says what is missing.

The unit is commented; read it before enabling it. Two things worth
knowing: it binds `127.0.0.1`, and it reads secrets from
`~/.config/fluxkrea/env` rather than from the config file, which is what
keeps `config.toml` committable and shareable across the fleet.

## A teaching node: one server, a room of students

The fleet case is one operator over an SSH tunnel. The lab case is
different: twenty people on the same LAN, each opening a URL, none of
them with a shell on the node. Two things make that work, and the first
one is the only place in this project that opens a port.

### Lab mode

```toml
# config.toml on the node
[dataset]
roots = ["/srv/student-data"]      # not optional here - see below

[daemon]
host      = "0.0.0.0"
port      = 8471
lab_mode  = true
```

or, for a session started by hand on the day:

```bash
fk serve --lab
```

`--lab` implies `--host 0.0.0.0` and turns lab mode on for that run
without editing anything.

**What lab mode actually trades.** The daemon normally refuses to bind
beyond loopback without `FLUXKREA_TOKEN`, because this API launches
processes and rewrites dataset folders — an open port on it is remote
code execution scoped to the node. Lab mode drops the token requirement
and puts the network in its place. That is a reasonable trade on an
isolated teaching VLAN and an unreasonable one anywhere a guest device
can associate. It is a config-file setting rather than an API one on
purpose: it cannot be turned on by anything arriving over the wire.

**`dataset.roots` is mandatory in lab mode and the daemon will refuse to
start without it.** With no roots the path check is a no-op, and the
roots are the only thing left scoping what the API can read and write.
Point them at the folder the students' data lives under and nothing else.

**The daemon prints where it is reachable**, every start, before uvicorn
says anything:

```
  ** listening beyond localhost **
     lab mode: no token required, anyone on this network can drive this node
     roots     /srv/student-data
     students  http://10.0.4.21:8471
```

That line is the student-facing instruction. An open port nobody
remembers opening is the failure this is arranged to prevent.

### Projects

There are no accounts and no passwords. A **project** — a named group of
dataset folders sharing one training configuration — is the identity.
The first thing a browser sees is a prompt to open one or create one, the
choice is remembered in that browser, and every screen is scoped to it.

That matters for the queue. One node trains one run at a time per GPU, so
a class of twenty is a queue, and the queue interleaves by project: each
project's next run goes before any project's one after that. A student who
submits five variations at nine in the morning does not hold the card until
lunch, and everybody's first run gets on early. Within a round it is still
first-come-first-served.

Set a room up from the node's own terminal rather than from eight
browsers:

```bash
for bench in 1 2 3 4 5 6 7 8; do
  fk dataset register "/srv/student-data/bench-$bench"
  fk projects new "Bench $bench" --dataset "bench-$bench"
done
fk projects list
```

`fk projects show <id>` prints its datasets, its shared training config and
its runs with queue positions. `fk jobs list` shows the whole waiting order
with a project against each row; `fk jobs list --project <id>` narrows the
rows and deliberately not the waiting order underneath them.

Projects are not authenticated — they are a name a browser chose. This is
fairness between good-faith parties, not an anti-abuse control. Nothing
stops somebody typing a new project name per run, and on a lab network
that is a conversation rather than a security boundary.

### Getting the LoRA off the node

A finished run offers **Download** and **Publish to ComfyUI** in the
monitor. Publishing copies the `.safetensors` into the node's own ComfyUI
install, in the folder the model belongs to:

| | |
|---|---|
| `flux2`, `flux2-klein-4b`, `flux2-klein-9b` | `models/loras/flux2` |
| `krea2` | `models/loras/krea2` |
| `flux1` | `models/loras/flux1` |

The folder comes off the model record (`Model.lora_dir`), never from the
run or file name — a Krea 2 LoRA in the FLUX.2 folder loads, generates
noise, and reads as a bad training run rather than a misfiled file.
Publishing needs `backends.comfyui_path` set; without it the button says
so rather than failing on click. An existing file of the same name is
never silently replaced: on a shared node it is somebody else's afternoon.

## The laptop

Nothing to install on the nodes beyond the above. Open a tunnel per node:

```bash
ssh -N -L 8471:localhost:8471 you@olympus-1
ssh -N -L 8472:localhost:8472 you@olympus-2
```

Then write `~/.config/fluxkrea/fleet.toml` (or `%APPDATA%\FluxKrea\fleet.toml`
on Windows):

```toml
[[node]]
name = "olympus-1"
url  = "http://localhost:8471"

[[node]]
name = "olympus-2"
url  = "http://localhost:8472"
```

and check it:

```bash
fk fleet status
fk fleet datasets
```

## Windows

`fk serve` runs as a plain console process. There is no `.bat`-only entry
point anywhere in this project, and nothing assumes a drive letter or a
path separator — the same commands work on both platforms.
