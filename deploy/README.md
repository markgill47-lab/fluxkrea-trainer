# Deployment

## A fleet node (Linux)

```bash
pip install -e ".[daemon]"
cp deploy/fluxkrea.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fluxkrea
systemctl --user status fluxkrea
```

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
