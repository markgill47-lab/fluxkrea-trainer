# Deployment

## A fleet node (Linux)

```bash
pip install -e ".[daemon]"
cp deploy/fluxkrea.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fluxkrea
systemctl --user status fluxkrea
```

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
